import os
from tqdm import tqdm
labels = ('Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise')
'''
for label in labels:
    for count in tqdm(range(1, 1001)):
        os.system('cp /media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/train/{}/{}.png \
            /media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/picked_dataset/train/{}/{}.png'.format(label, str(count).zfill(4), label, str(count).zfill(4)))
'''
for label in labels:
    for count in tqdm(range(1, 501)):
        os.system('cp /media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/test/{}/{}.png \
            /media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/picked_dataset/test/{}/{}.png'.format(label, str(count).zfill(4), label, str(count).zfill(4)))