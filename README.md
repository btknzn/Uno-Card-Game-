In this project, our goal is detecting Uno Card via computer camera. When we are doing this project we used OPENCV, KERAS, SKLEARN, NUMPY, H5PY LİBRARY, Pandas, Matplotlib.

In Unocarddetector.py, you can see our unocard detection code and in cnn.py , how we find weight for unocard detection code. Unocarddetector contains two main
system one of them is ımage preprocessing which finds the image in area the other one ise decision system, which decides which card is this. Main goal why we divided parts and 
did not apply our system cnn is that we want to increase succes of our system and cnn gives us neural network weights for Unocarddettector.pys' decision system.
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
