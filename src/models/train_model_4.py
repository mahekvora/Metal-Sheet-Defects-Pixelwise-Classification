#MODEL 4
import sys
import tensorflow

from small_data_generator import *
from feature_engineering import *
from tensorflow import keras
from keras.models import Sequential
train_df['ImageId']=train_df['ImageId'].map(lambda element: element[:-4]+'.npy')
val_df['ImageId']=val_df['ImageId'].map(lambda element: element[:-4]+'.npy')
print("Train Data Shape :", train_df.shape, "Val Data Shape :", val_df.shape)
partition_small={'train':train_df['ImageId'].tolist(),'validation':val_df['ImageId'].tolist()}
labels_small=dict(zip(df_small.ImageId,df_small.Any_Class))
# Parameters
params = {'dim': (256,1600),
          'batch_size': 16,
          'n_classes': 2,
          'n_channels': 3,
          'shuffle': True}

# Generators
training_generator = DataGenerator(partition_small['train'], labels_small, **params)
validation_generator = DataGenerator(partition_small['validation'], labels_small, **params)


# Design model
model1 = tensorflow.keras.Sequential()
model1.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(256, 1600, 3)))
model1.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model1.add(keras.layers.MaxPooling2D((2, 2)))
model1.add(keras.layers.Flatten())
model1.add(keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform'))
model1.add(keras.layers.Dense(2, activation='softmax'))
# compile model
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train model on dataset
history=model1.fit(training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    epochs=30,
                   )
new_model = model1
