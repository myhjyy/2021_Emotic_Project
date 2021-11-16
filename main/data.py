from PIL import Image
import face_recognition
import os
import os.path
import numpy as np
import json
from tqdm import tqdm

test_img = []
test_label = []
train_data_num = 500
test_data_num = 250
# You should input your dataset path
dataset_path = "/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/" # your own data path that contains 'main/' directory

# manage train dataset -------------------------------------------------------------- #
class train_data():
    def __init__(self):
        super().__init__()
        self.train_img = []
        self.train_skel = []
        self.train_label = []
        self.error_person = {}
        self.error_count = 0
        self.data_count = {}
        self.labels = ('Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise')
        for label in self.labels:
            self.error_person[label] = []
        
    def load(self):
        print("Start loading train data")
        for i, label in enumerate(self.labels):
            print(label, "in train file")
            for count in tqdm(range(1, train_data_num+1)):
                image = face_recognition.load_image_file(dataset_path + "dataset/train/{}/{}.png".format(label, str(count).zfill(4)))
                face_locations = face_recognition.face_locations(image)
                if (len(face_locations)) != 1:
                    self.error_person[label].append(count)
                    self.error_count += 1
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
        print("not one person detected in train images :", self.error_count)
        print("trained data :", self.data_count)
        print("Reshaping train input...")
        tmp_train = []
        for i, item in enumerate(self.train_img):
            tmp_train = np.append(tmp_train, np.array([[np.reshape(item, (3, 94, 94), order='F')]]))
        tmp_train = np.reshape(tmp_train, (-1, 3, 94, 94))
        print("Reshaping train input is done!")
        return tmp_train, self.train_label
        # return 1)output face image in numpy, 2)output label in numpy

    def save(self, new = None):
        if (new == None):
            if os.path.isfile(dataset_path + "dataset/numpy_data/X_train.npy") and os.path.isfile(dataset_path + "dataset/numpy_data/Y_train.npy"):
                print("Load train data from saved numpy file...")
                X_train = np.load(dataset_path + "dataset/numpy_data/X_train.npy")
                Y_train = np.load(dataset_path + "dataset/numpy_data/Y_train.npy")
                print("Train load success!")
                return X_train, Y_train
            else :
                print("There is no saved train numpy file. Make new numpy file...")
        elif (new != "new"):
            print("Word is not same with \'new\'!! You should use \'new\' or just blank!!")
            exit()
        else:
            if os.path.isfile(dataset_path + "dataset/numpy_data/X_train.npy"):
                os.remove(dataset_path + "dataset/numpy_data/X_train.npy")
            if os.path.isfile(dataset_path + "dataset/numpy_data/Y_train.npy"):
                os.remove(dataset_path + "dataset/numpy_data/Y_train.npy")

        X_train, Y_train = self.load()
        np.save(dataset_path + "dataset/numpy_data/X_train.npy", X_train)
        np.save(dataset_path + "dataset/numpy_data/Y_train.npy", Y_train)
        return X_train, Y_train

    def skel(self, save = None, new = None):
        if save == "save":
            if not os.path.isfile(dataset_path + "dataset/numpy_data/X_skel_train.npy"):
                print("There is no saved skel-train numpy file. Make new numpy file...")
            else:
                if new == "new":
                    print("Make new skel-train numpy file...")
                    if os.path.isfile(dataset_path + "dataset/numpy_data/X_skel_train.npy"):
                        os.remove(dataset_path + "dataset/numpy_data/X_skel_train.npy")
                else:
                    print("Load skel-train data from saved numpy file...")
                    X_train_skel = np.load(dataset_path + "dataset/numpy_data/X_skel_train.npy", self.train_skel)
                    print("Load success!")
                    return X_train_skel
        
        self.train_skel = []
        for label in self.labels:
            for count in range(1, train_data_num+1):
                contain = False
                for error in self.error_person[label]:
                    if error == count:
                        contain = True
                if contain:
                    continue
                with open(dataset_path+"dataset/train_keypoints/{}/{}_keypoints.json".format(label, str(count).zfill(4)), "r") as f:
                    json_data = json.load(f)
                    json_data = np.reshape(np.array(json_data['people'][0]['pose_keypoints_2d']), (3, 25))
                    self.train_skel = np.append(self.train_skel, json_data)
        X_train_skel = np.reshape(self.train_skel, (-1, 3, 25))
        if save == "save":
            np.save(dataset_path + "dataset/numpy_data/X_skel_train.npy", X_train_skel)
        print("Make new file success!")
        return X_train_skel

# manage test dataset -------------------------------------------------------------- #
class test_data():
    def __init__(self):
        super().__init__()
        self.test_img = []
        self.test_skel = []
        self.test_label = []
        self.error_person = {}
        self.error_count = 0
        self.labels = ('Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise')
        for label in self.labels:
            self.error_person[label] = []

    def load(self):
        print("Start loading test data")
        for i, label in enumerate(self.labels):
            print(label, "in test file")
            #for count in tqdm(range(1, len(os.listdir("../dataset/test/{}".format(label)))+1)):
            for count in tqdm(range(1, test_data_num+1)):
                image = face_recognition.load_image_file(dataset_path + "dataset/test/{}/{}.png".format(label, str(count).zfill(4)))
                face_locations = face_recognition.face_locations(image)
                if (len(face_locations)) != 1:
                    self.error_person[label].append(count)
                    self.error_count += 1
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
        print("not one person detected in test images :", self.error_count)
        print("Reshaping test input...")
        tmp_test = []
        for i, item in enumerate(self.test_img):
            tmp_test = np.append(tmp_test, np.array([[np.reshape(item, (3, 94, 94), order='F')]]))
        tmp_test = np.reshape(tmp_test, (-1, 3, 94, 94))
        print("Reshaping test input is done!")
        return tmp_test, self.test_label
        # return 1)output face image in numpy, 2)output label in numpy

    def save(self, new = None):
        if (new == None):
            if os.path.isfile(dataset_path + "dataset/numpy_data/X_test.npy") and os.path.isfile(dataset_path + "dataset/numpy_data/X_test.npy"):
                print("Load test data from saved numpy file...")
                X_test = np.load(dataset_path + "dataset/numpy_data/X_test.npy")
                Y_test = np.load(dataset_path + "dataset/numpy_data/Y_test.npy")
                print("Test load success!")
                return X_test, Y_test
            else :
                print("There is no saved test numpy file. Make new numpy file...")
        elif (new != "new"):
            print("Word is not same with \'new\'!! You should use \'new\' or just blank!!")
            exit()
        else:
            if os.path.isfile(dataset_path + "dataset/numpy_data/X_test.npy"):
                os.remove(dataset_path + "dataset/numpy_data/X_test.npy")
            if os.path.isfile(dataset_path + "dataset/numpy_data/Y_test.npy"):
                os.remove(dataset_path + "dataset/numpy_data/Y_test.npy")

        X_test, Y_test = self.load()
        np.save(dataset_path + "dataset/numpy_data/X_test.npy", X_test)
        np.save(dataset_path + "dataset/numpy_data/Y_test.npy", Y_test)
        return X_test, Y_test

    def skel(self, save = None, new = None):
        if save == "save":
            if not os.path.isfile(dataset_path + "dataset/numpy_data/X_skel_test.npy"):
                print("There is no saved skel-test numpy file. Make new numpy file...")
            else:
                if new == "new":
                    print("Make new skel-test numpy file...")
                    if os.path.isfile(dataset_path + "dataset/numpy_data/X_skel_test.npy"):
                        os.remove(dataset_path + "dataset/numpy_data/X_skel_test.npy")
                else:
                    print("Load skel-test data from saved numpy file...")
                    X_skel_test = np.load(dataset_path + "dataset/numpy_data/X_skel_test.npy", self.test_skel)
                    print("Load success!")
                    return X_skel_test
        
        self.test_skel = []
        for label in self.labels:
            for count in range(1, test_data_num+1):
                contain = False
                for error in self.error_person[label]:
                    if error == count:
                        contain = True
                if contain:
                    continue
                with open(dataset_path+"dataset/test_keypoints/{}/{}_keypoints.json".format(label, str(count).zfill(4)), "r") as f:
                    json_data = json.load(f)
                    json_data = np.reshape(np.array(json_data['people'][0]['pose_keypoints_2d']), (3, 25))
                    self.test_skel = np.append(self.test_skel, json_data)
        X_skel_test = np.reshape(self.test_skel, (-1, 3, 25))
        if save == "save":
            np.save(dataset_path + "dataset/numpy_data/X_skel_test.npy", X_skel_test)
        print("Make new file success!")
        return X_skel_test
#-------------------------------------------------------------------------------- #

# for check img file
''' top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    print(type(pil_image))
    pil_image.show()'''