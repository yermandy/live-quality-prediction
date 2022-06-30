import torch
import torch.backends.cudnn as cudnn
from data import cfg_mnet
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
import numpy as np
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from layers.functions.prior_box import PriorBox


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.device(f"cuda:0")
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


torch.set_grad_enabled(False)
cfg = cfg_mnet
# net and model
net = RetinaFace(cfg=cfg_mnet, phase = 'test')
weights_path = 'weights/mobilenet0.25_Final.pth'
net = load_model(net, weights_path, load_to_cpu=True)
net.eval()
print('Finished loading model!')
# print(net)
cudnn.benchmark = True
device = torch.device("cpu")
net = net.to(device)

_t = {'forward_pass': Timer(), 'misc': Timer()}


def detect(img_raw):
    # img_raw = cv2.imread(full_image_path, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)

    # testing scale
    target_size = 1600
    max_size = 2150
    
    height, width, channels = img_raw.shape
    img_size_min = min(height, width)
    img_size_max = max(height, width)
    
    resize = float(target_size) / float(img_size_min)

    # prevent bigger axis from being more than max_size:
    if np.round(resize * img_size_max) > max_size:
        resize = float(max_size) / float(img_size_max)
    origin_size = True
    if origin_size:
        resize = 1

    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    # forward pass
    _t['forward_pass'].tic()
    loc, conf, landms = net(img)  
    _t['forward_pass'].toc()
    
    # decode 
    _t['misc'].tic()
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    confidence_threshold = 0.5
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    top_k = 10
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    nms_threshold = 0.1
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    dets = np.concatenate((dets, landms), axis=1)
    _t['misc'].toc()

    bounding_boxes = []
    landmarks = []

    for det in dets:
        
        x1, y1, x2, y2 = [int(det[d]) for d in range(4)]
        confidence = det[4]

        bounding_boxes.append([x1, y1, x2, y2, confidence])
        landmarks.append([int(det[d]) for d in range(5, 15)])

    return np.array(bounding_boxes), np.array(landmarks)