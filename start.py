import cv2, time
from skimage import transform as trans
import numpy as np
from detection import detect
from model.cnn_fq import model
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

  
# define a video capture object 
capture = cv2.VideoCapture(0) 

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# capture.set(3, 200)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
# capture.set(4, 200)

ret, frame = capture.read()

scale_percent = 40
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)


destination = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041]], dtype=np.float32 )

destination[:,0] -= 30
destination[:,1] -= 51

center = np.array([[width / 2, height / 2]], dtype=np.float32)

destination += center


## quality prediction model
net = model(cuda=0, checkpoint_path='results/checkpoints/checkpoint.pt')


def get_affine_transform_matrix(src_pts, dst_pts):
    tfm = np.float32([[1, 0, 0], [0, 1, 0]])
    n_pts = src_pts.shape[0]
    ones = np.ones((n_pts, 1), src_pts.dtype)
    src_pts_ = np.hstack([src_pts, ones])
    dst_pts_ = np.hstack([dst_pts, ones])

    A, res, rank, s = np.linalg.lstsq(src_pts_, dst_pts_)

    if rank == 3:
        tfm = np.float32([
            [A[0, 0], A[1, 0], A[2, 0]],
            [A[0, 1], A[1, 1], A[2, 1]]
        ])
    elif rank == 2:
        tfm = np.float32([
            [A[0, 0], A[1, 0], 0],
            [A[0, 1], A[1, 1], 0]
        ])

    return tfm


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


def preprocess_photo(photo_array, bbox, scale=0.6):
    # frame
    # path  = self.images[index]
    # bb    = self.bbs[index]
    # image = Image.open(join(self.path_to_images, path))
    # image = image.convert('RGB')

    x1, y1, x2, y2 = bbox[0:4]
    w_scale = ((x2 - x1) * scale) / 2
    h_scale = ((y2 - y1) * scale) / 2
    x1 -= int(w_scale)
    y1 -= int(h_scale)
    x2 += int(w_scale)
    y2 += int(h_scale)

    photo_array = cv2.cvtColor(photo_array, cv2.COLOR_BGR2RGB)
    pil_img = transforms.ToPILImage()(photo_array)
    pil_img = pil_img.crop((x1, y1, x2, y2))
    tensor = transform(pil_img)

    pil_img = transforms.ToPILImage()(tensor)
    cv2.imshow('photo', np.array(pil_img)) 
    
    return tensor.unsqueeze(0)


font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 50)
fontScale              = 1
fontColor              = (0,0,255)
lineType               = 2

qualities = [0] * 100
qualities_mean = [0] * 100

while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = capture.read()

    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    frame = cv2.flip(frame, 1)

    bboxes, landmarks = detect(frame)

    # print(bbox)
    if len(bboxes) > 0:
        bbox = bboxes[0].astype(np.int)
        lm = landmarks[0].astype(np.int)

        x = preprocess_photo(frame, bbox)
        quality = net(x).squeeze().item()
    
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

        cv2.circle(frame, (lm[0], lm[1]), 1, (0, 0, 255), 4)
        cv2.circle(frame, (lm[2], lm[3]), 1, (0, 255, 255), 4)
        cv2.circle(frame, (lm[4], lm[5]), 1, (255, 0, 255), 4)
        cv2.circle(frame, (lm[6], lm[7]), 1, (0, 255, 0), 4)
        cv2.circle(frame, (lm[8], lm[9]), 1, (255, 0, 0), 4)

        cv2.putText(frame, f'{quality:.4f}', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        
        
        qualities.pop(0)
        qualities.append(quality)
        
        plt.cla()
        plt.plot(qualities)
        
        N = 20
        mean = np.convolve(qualities[-N:], np.ones(N)/N, mode='valid')
        qualities_mean.pop(0)
        qualities_mean.append(mean)
        
        plt.plot(qualities_mean)
        
        plt.ylim(0, 1)
        # plt.show(qualities)
        plt.pause(.001)
        # plt.close()
    # Display the resulting frame 
    cv2.imshow('frame', frame) 
    # cv2.imwrite('detected.png', frame) 
    
      
    if cv2.waitKey(1) == 27:
        break
  
# After the loop release the cap object 
capture.release()
# Destroy all the windows 
cv2.destroyAllWindows()