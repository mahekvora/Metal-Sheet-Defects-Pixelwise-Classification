#MODEL 0
#command line arguments argv[1]=partition argv[2]=labels
import sys
import tensorflow
from data_generator import *
from feature_engineering import *
from tensorflow import keras
from keras.models import Sequential
train_df['ImageId']=train_df['ImageId'].map(lambda element: element[:-4]+'.npy')
val_df['ImageId']=val_df['ImageId'].map(lambda element: element[:-4]+'.npy')
print("Train Data Shape :", train_df.shape, "Val Data Shape :", val_df.shape)
partition={'train':train_df['ImageId'].tolist(),'validation':val_df['ImageId'].tolist()}
labels=dict(zip(df.ImageId,df.Any_Class))

# Parameters
params = {'dim': (256,1600),
          'batch_size': 16,
          'n_classes': 2,
          'n_channels': 3,
          'shuffle': True}

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

# Design model
model = tensorflow.keras.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(256, 1600, 3)))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform'))
model.add(keras.layers.Dense(2, activation='softmax'))
	# compile model
opt = tensorflow.keras.optimizers.SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# Train model on dataset
model.fit(training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False
                   )