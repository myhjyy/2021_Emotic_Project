import torch
from PIL import Image
import face_recognition
import os
import os.path
import sys
import numpy as np
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(str(device)+" is using!")

test_img = []
test_label = []
train_data_num = 1000
test_data_num = 500

# manage train dataset -------------------------------------------------------------- #
class train_data():
    def __init__(self):
        super().__init__()
        self.train_img = []
        self.train_label = []
        self.error_person = []
        self.data_count = {}
        self.labels = ('Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise')
        
    def save(self):
        argument = sys.argv
        for i, label in enumerate(self.labels):
            print(label, "in train file")
            #for count in tqdm(range(1, len(os.listdir("../dataset/train/{}".format(label)))+1)):
            for count in range(1, train_data_num):
                image = face_recognition.load_image_file("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/train/{}/{}.png".format(label, str(count).zfill(4)))
                face_locations = face_recognition.face_locations(image)
                if (len(face_locations)) != 1:
                    self.error_person.append("{}/{}.png".format(label, str(count).zfill(4)))
                else:
                    top, right, bottom, left = face_locations[0]
                    face_image = image[top:bottom, left:right]
                    face_image = Image.fromarray(face_image)
                    face_image_resize = face_image.resize((94, 94))
                    face_image_resize = np.reshape(np.array(face_image_resize.getdata()), (94, 94, 3))
                    self.train_img.append(face_image_resize)
                    self.train_label.append(i)
                    if label not in self.data_count:
                        self.data_count[label] = 1
                    else:
                        self.data_count[label] += 1
        print("train data processing is done.")
        self.train_img = np.array(self.train_img)
        self.train_label = np.array(self.train_label)
        self.data_count = np.array(self.data_count)
        print("not one person detected in train images :", len(self.error_person))
        print("trained data :", self.data_count)
        np.save("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/train/X_train_init", self.train_img)
        np.save("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/train/Y_train", self.train_label)
        return self.train_img, self.train_label, self.data_count
        # return 1)output face image in numpy, 2)output label in numpy, 3)output number of labeled data in numpy

# manage test dataset -------------------------------------------------------------- #
class test_data():
    def __init__(self):
        super().__init__()
        self.test_img = []
        self.test_label = []
        self.error_person = []
        self.labels = ('Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise')

    def save(self):
        for i, label in enumerate(self.labels):
            print(label, "in test file")
            #for count in tqdm(range(1, len(os.listdir("../dataset/test/{}".format(label)))+1)):
            for count in range(1, test_data_num):
                image = face_recognition.load_image_file("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/test/{}/{}.png".format(label, str(count).zfill(4)))
                face_locations = face_recognition.face_locations(image)
                if (len(face_locations)) != 1:
                    self.error_person.append("{}/{}.png".format(label, str(count).zfill(4)))
                else:
                    top, right, bottom, left = face_locations[0]
                    face_image = image[top:bottom, left:right]
                    face_image = Image.fromarray(face_image)
                    face_image_resize = face_image.resize((94, 94))
                    face_image_resize = np.reshape(np.array(face_image_resize.getdata()), (94, 94, 3))
                    self.test_img.append(face_image_resize)
                    self.test_label.append(i)
        print("test data processing is done.")
        self.test_img = np.array(self.test_img)
        self.test_label = np.array(self.test_label)
        print("not one person detected in test images :", len(self.error_person))
        np.save("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/test/X_test_init", self.test_img)
        np.save("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/test/Y_test", self.test_label)
        return self.test_img, self.test_label
        # return 1)output face image in numpy, 2)output label in numpy

#-------------------------------------------------------------------------------- #

# for check img file
''' top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    print(type(pil_image))
    pil_image.show()'''