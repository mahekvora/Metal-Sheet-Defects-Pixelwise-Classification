#!/usr/bin/env python
# argv[1]=training (boolean variable for training or not)
# argv[2]=testing (boolean variable for testing or not) ##
# argv[3]=Resume (boolean variable for resuming) 
# argv[4]=device (with or without GPU) ##
# argv[5]=dataset_path (path to the dataset)
# argv[6]=num epochs
# argv[7]=classification model (select one of multiple models) 
# argv[8]=input size of images to model (height)
# argv[9]=input size of images to model (width)
# argv[10]=batch size 
# argv[11]=learning rate 
import sys
import tensorflow
import numpy as np
# print(sys.argv[1])
# print(sys.argv[2])
# print(sys.argv[3])
# print(sys.argv[4])
# print(sys.argv[5])
# print(sys.argv[6])
# print(sys.argv[7])
# print(sys.argv[8])
# print(sys.argv[9])
# print(sys.argv[10])
# print(sys.argv[11])
from small_data_generator import *
from models import *
from tensorflow import keras
from keras.models import Sequential
from keras.models import load_model

train_df['ImageId']=train_df['ImageId'].map(lambda element: element[:-4]+'.npy')
val_df['ImageId']=val_df['ImageId'].map(lambda element: element[:-4]+'.npy')
print("Train Data Shape :", train_df.shape, "Val Data Shape :", val_df.shape)
partition_small={'train':train_df['ImageId'].tolist(),'validation':val_df['ImageId'].tolist()}
labels_small=dict(zip(df_small.ImageId,df_small.Any_Class))
# Parameters
params = {'dim': (int(sys.argv[8]),int(sys.argv[9])),
          'batch_size': int(sys.argv[10]),
          'n_classes': 2,
          'n_channels': 3,
          'shuffle': True}

# Generators
training_generator = DataGenerator(partition_small['train'], labels_small, **params)
validation_generator = DataGenerator(partition_small['validation'], labels_small, **params)


# Design model
model1 = VGG_model()


# Train model on dataset
if int(sys.argv[1])==1:
    history=model1.fit(training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    epochs=int(sys.argv[6]),
                   )
if int(sys.argv[3])==1:
    model = keras.models.load_model(sys.argv[5]+'/updated_model_4_small_data.h5',compile=True)
    history=model.fit(training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    epochs=int(sys.argv[6]),
                    initial_epoch=30
                   )
    model.save(sys.argv[5]+'/'+'epochs'+sys.argv[6]+'_model_4_small_data.h5')
#val-acc val-loss training-acc training-loss lr epoch opt

if int(sys.argv[2])==1:
    model = keras.models.load_model(sys.argv[5]+'/updated_model_4_small_data.h5',compile=True)
    test_list=[]
    from numpy import save
    dir=os.listdir(sys.argv[5]+'/test_images_np/')
    dir.sort()
    for ID in dir:
        test_list.append(ID)
    dim=(256,1600)
    X_test = np.empty((16, *dim, 3))
    for i,ID in enumerate(test_list[:16]):
        X_test[i,] = np.load(sys.argv[5]+'/test_images_np/' + ID)
        #print(i)
    predictions=model.predict(X_test)
    ans1=[]
    for i in range(len(predictions)):
        ans1.append(np.argmax(predictions[i]))
    print(ans1)



