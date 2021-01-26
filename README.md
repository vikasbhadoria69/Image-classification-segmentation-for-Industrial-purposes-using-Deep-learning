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

***Problems in Phase 1: No Problems***

### Phase 2: Using transfer learning to deploy Microsoft’s ResNet model to observe the power of CNN for image classification. 
Here the model will classify which images are normal & which are with defects.

* **ResNet:** This is the model proposed by Microsoft which got 96.4% accuracy in the ImageNet
2016 competition. ResNet is used as a pre-trained model for several applications. 
* After training the model for 40 epochs an accuracy of 87% on the test images was achieved. Below is the confusion matrix of the result after testing the model on new images.

![alt text](https://github.com/vikasbhadoria69/Image-classification-segmentation-for-Industrial-purposes-using-Deep-learning/blob/main/Images/imgcon.png)

* The classification report gave really good results for precision and recall.

![alt text](https://github.com/vikasbhadoria69/Image-classification-segmentation-for-Industrial-purposes-using-Deep-learning/blob/main/Images/img15.jpg)

***Problems in Phase 2: No Problems***

### Phase 3: is about image segmentation using the ResUNet model. localized the defects in a defective image.
After the classification has been done the first task was achieved. In this phase the focus is on task 2 which is ‘image segmentation’. The defect in the defective images will
belocalized. For this task the model used is ResUNet.
* For classical CNNs: they are generally used when the entire image is needed to be classified as a class label.
* For Unet: pixel level classification is performed. U-net formulates a loss function for every pixel in the input image. Softmax function is applied to every pixel which makes the segmentation problem work as a classification problem where classification is performed on every pixel of the image.
* The ResUNet model has been trained on **40 epochs** for this project.

***Problems in Phase 3: No Problems***

### Phase 4: describes how good the model’s predictions are.
In this phase the model built and trained is put to test on images which it has
never seen before. The test images are random images from the dataset. The task of the
trained model is to classify these images as defective or normal and then take all the
defective images and mark the defect using its predicted mask.
***The prediction of the model can be seen below. On the left side are the real images(green)
and on the right side are the images predicted(red) using the trained model.***

![alt text](https://github.com/vikasbhadoria69/Image-classification-segmentation-for-Industrial-purposes-using-Deep-learning/blob/main/Images/image%20(2).png)
![alt text](https://github.com/vikasbhadoria69/Image-classification-segmentation-for-Industrial-purposes-using-Deep-learning/blob/main/Images/image%20(3).png)
![alt text](https://github.com/vikasbhadoria69/Image-classification-segmentation-for-Industrial-purposes-using-Deep-learning/blob/main/Images/image%20(6).png)
![alt text](https://github.com/vikasbhadoria69/Image-classification-segmentation-for-Industrial-purposes-using-Deep-learning/blob/main/Images/image%20(7).png)
![alt text](https://github.com/vikasbhadoria69/Image-classification-segmentation-for-Industrial-purposes-using-Deep-learning/blob/main/Images/image%20(8).png)

***Problems in Phase 4: In most of the cases the trained model is performing really good to localize the defect. But
there are few errors as seen in the 4th prediction. The type of error predicted by the model is
3 but in reality its type 1. This can be due to the imbalance dataset which included defect type
3 images the most***

## Conclusions
In this project a pipeline has been created for the entire project which can take in the input and classify the image as normal
or defected and later the defected images are put as an input to the segmentation model
which predicts the location of the defect.
The overall results can be concluded in 3 points:
* **The model works pretty good to classify the images as normal or defective.**
* **The model is well accurate to detect the mask of defective image which locates the defects accurately.**
* **The model is producing some inappropriate results in case of detecting the type of defect. As there are 4 types of defects in the dataset. The model is more biased to predict type 3 defects. This can be due to the imbalance dataset as images with type 3 defects are the most in number. With a well balanced dataset this issue can be solved. This can be achieved via image augmentation which can be a future scope for this project.**

