In this project, our goal is detecting Uno Card via computer camera. When we are doing this project we used OPENCV, KERAS, SKLEARN, NUMPY, H5PY LİBRARY, Pandas, Matplotlib.

Unocarddetector contains two main systems one of them is ımage preprocessing which finds the image in area the another one is a decision system, which decides the card number and color. For decision system, we trained neural network. The aim of preprocessing is increasing the success of Neural Network. 
In Unocarddetector.py, you can see our unocard detection code and in cnn.py , how we find weight for unocard detection code. In my system I first applied preprocessing part and then I planned to applying feature matching but every uno cards detected in my algorithm as 0 due to the small circle around the uno number. Because of that we trained neural network in cnn.py and it is taken this neural network shape and weight and added in  Uno card Detection code, which can detect card number.


What is SVHN? The Street View House Numbers (SVHN) is a real world image dataset used for developing machine
learning and object recognition algorithms. It is one of the commonly used benchmark datasets as It
requires minimal data preprocessing and formatting. Although it shares some similarities
with MNIST where the images are of small cropped digits, SVHN incorporates an order of magnitude
more labelled data (over 600,000 digit images). It also comes from a significantly harder real world
problem of recognising digits and numbers in nat ural scene images.
You can download KERAS SVHN DATASET:
https://www.kaggle.com/olgabelitskaya/svhn-digit-recognition/data
Below you can see flow chart of my code:
![alt text](https://github.com/btknzn/Uno-Card-Game-/blob/master/FLOWCART.PNG)
 For better understanding you can read my project report:
 https://github.com/btknzn/Uno-Card-Game-/blob/master/UNOCARDGAME.pdf
 
 Below you can see projects steps:
 First take Image from camera:
 
![alt text](https://github.com/btknzn/Uno-Card-Game-/blob/master/1.PNG)

Seperate the Image:
After taking Image from camera, our code separate to image 8 different areas

![alt text](https://github.com/btknzn/Uno-Card-Game-/blob/master/2.PNG)

Gaussian BLUR:
In Gaussian Blur operation, the image is convolved with a Gaussian filter instead of the box filter. The Gaussian filter is a low-pass filter that removes the high-frequency components are reduced.

![alt text](https://github.com/btknzn/Uno-Card-Game-/blob/master/3.PNG)

Threshold:
After that Otsu and binary threshold algorithms are applied.

![alt text](https://github.com/btknzn/Uno-Card-Game-/blob/master/4.PNG)

Canny Edge Detector:
After thresholding, we applied Canny Edge detection and we detected our systems’ edges

![alt text](https://github.com/btknzn/Uno-Card-Game-/blob/master/5.PNG)

Corner detection:
After corner detection, we detected edges via pre-written open cv library (goodFeaturesToTrack).

![alt text](https://github.com/btknzn/Uno-Card-Game-/blob/master/6.PNG)

Crop the number:

![alt text](https://github.com/btknzn/Uno-Card-Game-/blob/master/7.PNG)

Neural Network succes on SVHN TEST DATA SET:

![alt text](https://github.com/btknzn/Uno-Card-Game-/blob/master/8.PNG)

NUMBER DETECTİON PART( Outs are results, pictures are input)

![alt text](https://github.com/btknzn/Uno-Card-Game-/blob/master/9.PNG)
