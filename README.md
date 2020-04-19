In this project, our goal is detecting Uno Card via computer camera. When we are doing this project we used OPENCV, KERAS, SKLEARN, NUMPY, H5PY LİBRARY, Pandas, Matplotlib.

Unocarddetector contains two main systems one of them is ımage preprocessing which finds the image in area the other one ise decision system, which decides the card number and color. For decision system, we trained neural network. The aim of preprocessing is increasing the success of Neural Network. 
In Unocarddetector.py, you can see our unocard detection code and in cnn.py , how we find weight for unocard detection code. In my system I first applied preprocessing part and then I planned to applying feature matching but every uno cards detected in my algorithm as 0 due to the small circle around the uno number. Because of that we trained neural network in cnn.py and it is taken this neural network shape and weight and added in  Uno card Detection code, which can detect card number.

You can download KERAS DATASET:
https://www.kaggle.com/olgabelitskaya/svhn-digit-recognition/data
Below you can see flow chart of my code:
![alt text](https://github.com/btknzn/Uno-Card-Game-/blob/master/FLOWCART.PNG)
 For better understanding you can read my project report:
 https://github.com/btknzn/Uno-Card-Game-/blob/master/UNOCARDGAME.pdf
 
 Below you can see projects steps:
 First take Image from camera:
 
![alt text](https://github.com/btknzn/Uno-Card-Game-/blob/master/1.PNG)

Seperate the Image:

![alt text](https://github.com/btknzn/Uno-Card-Game-/blob/master/2.PNG)

Gaussian BLUR:

![alt text](https://github.com/btknzn/Uno-Card-Game-/blob/master/3.PNG)

Threshold:

![alt text](https://github.com/btknzn/Uno-Card-Game-/blob/master/4.PNG)

Canny Edge Detector:

![alt text](https://github.com/btknzn/Uno-Card-Game-/blob/master/5.PNG)

Corner detection:

![alt text](https://github.com/btknzn/Uno-Card-Game-/blob/master/6.PNG)

Crop the number:

![alt text](https://github.com/btknzn/Uno-Card-Game-/blob/master/7.PNG)

Neural Network succes on SVHN TEST DATA SET:

![alt text](https://github.com/btknzn/Uno-Card-Game-/blob/master/8.PNG)

NUMBER DETECTİON PART( Outs are results, pictures are input)

![alt text](https://github.com/btknzn/Uno-Card-Game-/blob/master/9.PNG)
