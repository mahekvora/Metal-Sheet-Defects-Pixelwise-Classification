import tensorflow
from tensorflow import keras
import sys
#from keras.models import Sequential
#from keras.models import load_model

def VGG_model():
    model1 = tensorflow.keras.Sequential()
    model1.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(int(sys.argv[8]), int(sys.argv[9]), 3)))
    model1.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model1.add(keras.layers.MaxPooling2D((2, 2)))
    model1.add(keras.layers.Flatten())
    model1.add(keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform'))
    model1.add(keras.layers.Dense(2, activation='softmax'))
    # compile model
    opt=keras.optimizers.Adam(learning_rate=float(sys.argv[11]))
    model1.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model1
