import numpy 
import pandas 

import glob

import matplotlib.pylab as plt
import matplotlib.cm as cm

import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential, Model
from keras.optimizers import SGD, RMSprop, Adam, Nadam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from keras.layers import Dense, Dropout, LSTM
from keras.layers import Activation, Flatten, Input, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D 
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D

def digit_to_categorical(data):
    n = data.shape[1]
    data_cat = numpy.empty([len(data), n, 11])    
    for i in range(n):
        data_cat[:, i] = to_categorical(data[:, i], num_classes=11)        
    return data_cat


def cnn_model():    
    model_input = Input(shape=(32, 32, 1))
    x = BatchNormalization()(model_input)
        
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(model_input)
    x = MaxPooling2D(pool_size=(2, 2))(x) 
    
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)    
    x = Dropout(0.25)(x)
    
    x = Conv2D(64, (3, 3), activation='relu')(x)       
    x = Conv2D(64, (3, 3), activation='relu')(x)    
    x = Dropout(0.25)(x)
    
    x = Conv2D(196, (3, 3), activation='relu')(x)    
    x = Dropout(0.25)(x)
              
    x = Flatten()(x)
    
    x = Dense(512, activation='relu')(x)    
    x = Dropout(0.5)(x)
    
    y1 = Dense(11, activation='softmax')(x)
    y2 = Dense(11, activation='softmax')(x)
    y3 = Dense(11, activation='softmax')(x)
    y4 = Dense(11, activation='softmax')(x)
    y5 = Dense(11, activation='softmax')(x)
    
    model = Model(input=model_input, output=[y1, y2, y3, y4, y5])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


train_images = pandas.read_csv('housenumbers/train_images.csv')
train_labels = pandas.read_csv('housenumbers/train_labels.csv')
test_images = pandas.read_csv('housenumbers/test_images.csv')
test_labels = pandas.read_csv('housenumbers/test_labels.csv')
extra_images = pandas.read_csv('housenumbers/extra_images.csv')
extra_labels = pandas.read_csv('housenumbers/extra_labels.csv')

train_images.ix[:10,:10]
train_labels.ix[:10,:]
train_images = train_images.ix[:,1:].as_matrix().astype('float32')
train_labels = train_labels.ix[:,1:].as_matrix().astype('int16')

test_images = test_images.ix[:,1:].as_matrix().astype('float32')
test_labels = test_labels.ix[:,1:].as_matrix().astype('int16')

extra_images = extra_images.ix[:,1:].as_matrix().astype('float32')
extra_labels = extra_labels.ix[:,1:].as_matrix().astype('int16')
print('Label: ', train_labels[100])
plt.imshow(train_images[100].reshape(32,32), cmap=plt.cm.bone)

x_train = numpy.concatenate((train_images.reshape(-1, 32, 32, 1),
                             test_images.reshape(-1, 32, 32, 1)),
                            axis=0)
y_train = numpy.concatenate((digit_to_categorical(train_labels),
                             digit_to_categorical(test_labels)),
                            axis=0)

x_valid = extra_images.reshape(-1, 32, 32, 1)
y_valid = digit_to_categorical(extra_labels)

n = int(len(x_valid)/2)
x_test, y_test = x_valid[:n], y_valid[:n]
x_valid, y_valid = x_valid[n:], y_valid[n:]

x_train.shape, x_test.shape, x_valid.shape, \
y_train.shape, y_test.shape, y_valid.shape

y_train_list = [y_train[:, i] for i in range(5)]
y_test_list = [y_test[:, i] for i in range(5)]
y_valid_list = [y_valid[:, i] for i in range(5)]

cnn_model = cnn_model()
cnn_checkpointer = ModelCheckpoint(filepath='weights.best.cnn.hdf5', 
                                   verbose=2, save_best_only=True)
cnn_history = cnn_model.fit(x_train, y_train_list, 
                            validation_data=(x_valid, y_valid_list), 
                            epochs=75, batch_size=128, verbose=2, 
                            callbacks=[cnn_checkpointer])

cnn_model.load_weights('weights.best.cnn.hdf5')
cnn_scores = cnn_model.evaluate(x_test, y_test_list, verbose=0)

print("CNN Model 1. \n")
print("Scores: \n" , (cnn_scores))
print("First digit. Accuracy: %.2f%%" % (cnn_scores[6]*100))
print("Second digit. Accuracy: %.2f%%" % (cnn_scores[7]*100))
print("Third digit. Accuracy: %.2f%%" % (cnn_scores[8]*100))
print("Fourth digit. Accuracy: %.2f%%" % (cnn_scores[9]*100))
print("Fifth digit. Accuracy: %.2f%%" % (cnn_scores[10]*100))

print(cnn_model.summary())

plt.figure(figsize=(14, 7))

plt.plot(cnn_history.history['val_dense_2_acc'][35:], label = 'First digit')
plt.plot(cnn_history.history['val_dense_3_acc'][35:], label = 'Second digit')
plt.plot(cnn_history.history['val_dense_4_acc'][35:], label = 'Third digit')
plt.plot(cnn_history.history['val_dense_5_acc'][35:], label = 'Fourth digit')
plt.plot(cnn_history.history['val_dense_6_acc'][35:], label = 'Fifth digit')

plt.legend()
plt.title('Accuracy');

a = cv2.imread('7.JPEG')
gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)
a=cv2.resize(gray,(32,32))
pred=cnn_model.predict(a.reshape(1, 32, 32, 1))