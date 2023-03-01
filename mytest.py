import argparse
import math
import time
import os

import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from models.pfld import PFLDInference

cudnn.benchmark = True
cudnn.determinstic = True
cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate(path_test,pfld_backbone,transform):
    pfld_backbone.eval()
    names = os.listdir(path_test)
    count = 0
    for name in names:
        path_image = os.path.join(path_test,name)
        img = cv2.imread(path_image)
        img = cv2.resize(img,(112,112))
        input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input = input.astype(np.float32) / 255.0
        input = np.expand_dims(input, 0)
        input = torch.Tensor(input.transpose((0, 3, 1, 2)))

        pfld_backbone = pfld_backbone.to(device)
        _,landmarks = pfld_backbone(input)
        landmarks = landmarks.cpu().detach().numpy()
        landmarks = landmarks.reshape(landmarks.shape[0], -1,2)  # landmark
        if args.show_image:
            pre_landmark = landmarks[0] * [112, 112]
            coor = []            
            for (x, y) in pre_landmark.astype(np.int32):
                cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
                coor.append(x)
                coor.append(y)
            vector1 = (float(coor[63*2]) - float(coor[60*2]),float(coor[63*2+1]) - float(coor[60*2+1]))
            vector2 = (float(coor[64*2]) - float(coor[60*2]),float(coor[64*2+1]) - float(coor[60*2+1]))
            mul = vector1[0]*vector2[0] + vector1[1]*vector2[1]
            length_vec1 = math.sqrt(vector1[0]*vector1[0] + vector1[1]*vector1[1])
            length_vec2 = math.sqrt(vector2[0]*vector2[0] + vector2[1]*vector2[1])
            div = mul/(length_vec1*length_vec2)
            angle = math.acos(div)*180 / math.pi
            close_eye = float(coor[64*2+1]) - float(coor[63*2+1])
            if close_eye < 0:
                angle = -1 * angle                
            print("Angle of eye: ",angle)
            if angle > min_angle and angle < max_angle:
                print("Nhắm mắt")
            else: 
                print("Mở mắt")
            img = cv2.resize(img, (400, 400))
            if count % 1 == 0:
               cv2.imshow("show_img.jpg", img)
               cv2.waitKey(0)
            count += 1
    


def main(args):
    checkpoint = torch.load(args.model_path, map_location=device)
    pfld_backbone = PFLDInference().to(device)
    pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])

    transform = transforms.Compose([transforms.ToTensor()])
    path_dir = args.test_dataset

    validate(path_dir, pfld_backbone,transform)


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_path',
                        default="checkpoint/snapshot/checkpoint.pth.tar",type=str)
    parser.add_argument('--test_dataset',
                        default='data/Crop',type=str)
    parser.add_argument('--show_image', default=True, type=bool)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    min_angle = 0
    max_angle = 12
    args = parse_args()
    main(args)
