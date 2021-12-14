import json
import cv2
import numpy as np
from PIL import Image
import face_recognition

dataset_path = "/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/"
lines = {0:[1,15,16], 1:[2,5,8], 2:[3], 3:[4], 4:[], 5:[6], 6:[7], 8:[9,12], 12:[13], 15:[17], 16:[18]}

for i in range(229, 230):
    num = str(i)
    label = 'Happy'
    with open(dataset_path+'dataset/train_keypoints/{}/{}_keypoints.json'.format(label, num.zfill(4)), 'r') as f:
        json_data = json.load(f)
    keypoints = json_data['people'][0]['pose_keypoints_2d']
    print(keypoints)
    color = (255,255,255)
    red_color = (0, 0, 255)
    image = face_recognition.load_image_file(dataset_path + 'dataset/train/{}/{}.png'.format(label, num.zfill(4)))
    image = image[:,:,::-1].copy()
    keypoint_cor = []

    face_locations = face_recognition.face_locations(image)
    top, right, bottom, left = face_locations[0]
    face_image = image[top:bottom, left:right]

    #cv2.imshow('image', face_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    for idx in range(25):
        x = keypoints[3*idx]
        y = keypoints[3*idx+1]
        keypoint_cor.append((int(x),int(y)))
        if x==0 and y==0:
            continue
        #cv2.putText(image, "{}".format(idx),(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    for dot in lines:
        if keypoint_cor[dot] == (0,0):
            continue
        for i in lines[dot]:
            if keypoint_cor[i] == (0,0):
                continue
            image = cv2.line(image, keypoint_cor[dot], keypoint_cor[i], red_color, 3)

    for idx in range(25):
        x = keypoints[3*idx]
        y = keypoints[3*idx+1]
        if x==0 and y==0:
            continue
        image = cv2.circle(image, (int(x),int(y)), 4, color, -1)

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()