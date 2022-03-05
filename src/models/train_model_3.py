#MODEL 3
#lr 0.0005
import sys
import tensorflow

from small_data_generator import *
from feature_engineering import *
from tensorflow import keras
from keras.models import Sequential
# import matplotlib.pyplot
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
model3 = tensorflow.keras.Sequential()
model3.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(256, 1600, 3)))
model3.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model3.add(keras.layers.MaxPooling2D((2, 2)))
model3.add(keras.layers.Flatten())
model3.add(keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform'))
model3.add(keras.layers.Dense(2, activation='softmax'))
	# compile model
opt = keras.optimizers.Adam(learning_rate=0.0005)
model3.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])



# Train model on dataset
history=model3.fit(training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    epochs=3,
                   )
# def summarize_diagnostics(history):
# 	# plot loss
# 	matplotlib.pyplot.subplot(211)
# 	matplotlib.pyplot.title('Cross Entropy Loss')
# 	matplotlib.pyplot.plot(history.history['loss'], color='blue', label='train')
# 	matplotlib.pyplot.plot(history.history['val_loss'], color='orange', label='test')
# 	# plot accuracy
# 	matplotlib.pyplot.subplot(212)
# 	matplotlib.pyplot.title('Classification Accuracy')
# 	matplotlib.pyplot.plot(history.history['accuracy'], color='blue', label='train')
# 	matplotlib.pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
# 	# save plot to file
# 	matplotlib.pyplot.savefig(PATH_TO_DATA+'/adam_lr2.png')
# 	matplotlib.pyplot.close()
