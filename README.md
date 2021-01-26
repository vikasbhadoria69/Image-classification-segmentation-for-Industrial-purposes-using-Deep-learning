# Image classification & segmentation for Industrial purposes using Deep learning.
### Leveraged the power of Deep learning and build a model that can be used for classifying & detecting defects in industrial images.
* **Image Classification**: Used Transfer Learning technique and deploy Microsoft's ‘ResNet’ deep learning architecture on the dataset to classify which images are having defects and which are normal.
* **Image Segmentation**: Once the images with defects were classified with high accuracy the second step taken is to localize the defects. Pointed out where exactly the defects in an image are and the type of defect using ResUNet deep learning model.
* Tested the model on the images which it has never seen before. By achieving a good accuracy on test data, the model can be deployed in real world industrial applications for classifying defective product images and segmenting the exact location of defect.

## Code and Resources Used
**Python Version:** 3.7  
**Packages:** pandas, numpy, TensorFlow, matplotlib, TransferLearning, Keras. 

## Keywords
Deep learning, Convolutional Neural Networks, Transfer Learning, ResNet,
Image segmentation, Image classification, ResUNet.

## Project overview
Nowadays, semantic segmentation is one of the key problems in the field of computer vision.
Looking at the big picture, semantic segmentation is one of the high-level tasks that paves
the way towards complete scene understanding. 
In this project, I have discussed how to use deep convolutional neural networks to do image
classification & segmentation. The project is practical based and works on classification &
segmentation of the industrial images from a steel manufacturing company. The project is
developed in Python using the deep neural framework of Keras & Tensorflow.

## Dataset
* The dataset used is downloaded from kaggle which contains **12600 images** that contain **4
types of defects**, along with their location in the steel surface. The location is nothing but the
mask with the exact location of the defect. The masks of images are encoded using
**[Run-length encoding](https://en.wikipedia.org/wiki/Run-length_encoding)** and for this project I will be using a helper function to convert RLE to a
mask which is of the exact size of the image.
* Images received from the dataset are separated into training, validation and testing dataset.
The training dataset is used to train the model followed by validation dataset which is used to
validate the model’s performance.
* **[Datalink for training the model](https://drive.google.com/drive/folders/1Xn8O6nWcfIx-7HYRPgxomRfHj3V4SgEF?usp=sharing)**

## Methodology
##### The project work was divided into the following 4 phases

### Phase 1: Is about exploring & visualization of the dataset.
After looking for the dataset and finding the data it was really important to explore
the dataset and visualize it. This is a crucial step in any project related to deep learning as it
gives a lot of information about the data such as missing values, imbalance data, unique
values and so on. I did some visualizations to explore the dataset.

* Number of defective images(orange) v/s normal images(blue) in the dataset
![alt text](https://github.com/vikasbhadoria69/Image-classification-segmentation-for-Industrial-purposes-using-Deep-learning/blob/main/Images/img11.png)

* Number of defects in a single image(most of the images are with single defect)
![alt text](https://github.com/vikasbhadoria69/Image-classification-segmentation-for-Industrial-purposes-using-Deep-learning/blob/main/Images/img12.png)

* Number of images per type of defect

![alt text](https://github.com/vikasbhadoria69/Image-classification-segmentation-for-Industrial-purposes-using-Deep-learning/blob/main/Images/img13.png)

* A defective image and its corresponding mask indicating the exact location of the defect
![alt text](https://github.com/vikasbhadoria69/Image-classification-segmentation-for-Industrial-purposes-using-Deep-learning/blob/main/Images/img15.png)

* Two defective images of defect type 3 & 4 respectively, combined along with their respective masks
![alt text](https://github.com/vikasbhadoria69/Image-classification-segmentation-for-Industrial-purposes-using-Deep-learning/blob/main/Images/img14.jpg)



