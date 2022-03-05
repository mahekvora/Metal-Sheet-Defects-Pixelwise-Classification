#MODEL 2
#command line arguments argv[1]=partition argv[2]=labels

import sys
from data_generator import *
from feature_engineering import *
import matplotlib
from matplotlib import pyplot
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
model2 = tensorflow.keras.Sequential()
model2.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(256, 1600, 3)))
model2.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model2.add(keras.layers.MaxPooling2D((2, 2)))
model2.add(keras.layers.Flatten())
model2.add(keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform'))
model2.add(keras.layers.Dense(2, activation='softmax'))
	# compile model
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()
 

# Train model on dataset
history=model2.fit(training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    epochs=10,
                   )
summarize_diagnostics(history)