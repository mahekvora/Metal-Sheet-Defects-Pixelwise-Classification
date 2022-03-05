import numpy as np
from feature_engineering import *
import sklearn
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow import keras

PATH="/content/drive/My Drive/Datasets/caliche/severstal-steel-defect-detection"
columns = ["ImageId", "Class_1", "Class_2", "Class_3", "Class_4", "Any_Class", "Sum_of_Defects"]
y_columns = ["Any_Class"]
df = img_class_data[columns]
#id in the form of npy
df['ImageId']=df['ImageId'].map(lambda element: element[:-4]+'.npy')
train_df, val_df = train_test_split(df, stratify=df[["Any_Class"]], test_size=0.20, random_state=42)
# train_df['ImageId']=train_df['ImageId'].map(lambda element: element[:-4]+'.npy')
# val_df['ImageId']=val_df['ImageId'].map(lambda element: element[:-4]+'.npy')
# print("Train Data Shape :", train_df.shape, "Val Data Shape :", val_df.shape)
# partition={'train':train_df['ImageId'].tolist(),'validation':val_df['ImageId'].tolist()}
# labels=dict(zip(df.ImageId,df.Any_Class))
class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=16, dim=(256,1600), n_channels=3,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            size=len(ID)
            X[i,] = np.load(PATH+'/train_images_np/' + ID)

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)