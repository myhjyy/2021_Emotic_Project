# Emotion Classification using Skeleton Data

## Abstract
When classifying emotions in an image, most machine learning recognizes faces from images first and interprets them into an artificial neural network to classify emotions. However, it is sometimes difficult to tell the difference between emotions without seeing the context of the situation together. In this study, context data will be represented using human skeleton and analyzed whether this content can increase accuracy in classifying emotions. For more information, Click here [Korean](https://early-dimple-bf7.notion.site/Emotic-Project-124628ee2b1a4c6bb61c99ddd0c90b20).

## Code Explaination
- `main/main.py`: Main python code executing train and test. Training  and testing will done together.
- `main/data.py`: Code that manage input dataset. It also make face_detection imgs and image's numpy file for fast testing.
- `main/model.py`: Contains main network of facial and skeletal encoding.
- `dataset/`: Folder that contains images.
- `dataset/()_keypoints/`: Folder that contains images' keypoints in `.json` format.
- `dataset/numpy_data`: For faster training and testing, save information about the images.

## How To Run
For execute, the code needs two input argument; network type(S or F) and data input type(load or save).
- Network type: `S` means skeletal context encoding, and `F` means facial encoding.
- Data Input type: `load` means load data from input images and not use numpy file, `save` means load data from numpy file and if there are not such numpy file, make new numpy file. I recommend you to use numpy file for faster training and testing.

1. You have to modify `dataset_path` in `main/data.py`. The path has to contain `main/` folders.
2. Install env: `pip install -r main/requirements.txt`.
3. If you have your own data, go to number 4. If you use the data that we provide, go to `For Train and Test` section.
4. Input your dataset in `dataset` file. The dataset shoud splited into `dataset/train/` and `dataset/test/` files, and each dataset also has to splited with 7 emotion`(Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise)` annotations. `ex)dataset/train/Anger/0003.png`
5. Execute `python main/main.py --net F --data save`. Then, the `data.py` make you `dataset/` folder.
6. You should make images' skeleton data in dataset with Opensource:[Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).
7. Input the skeleton .json files with this format: `dataset/[test or train]_keypoints/[emotion label]/[num of images]_keypoints.json`. `ex) dataset/test_keypoints/Anger/0003_keypoints.json` The numbers of keypoints.json file and image should be same.

### For Train and Test
If you want to use new input images, make `dataset/` file and input images to test/, train/ split files. Then, just execute this:
``` python main/main.py --net [S or F] --data save --new True```

If you have datasets that I provide, then execute this:
``` python main/main.py --net [S or F] --data save```
