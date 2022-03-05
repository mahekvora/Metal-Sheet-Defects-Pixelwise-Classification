import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from google.colab.patches import cv2_imshow
#DATA CLEANING
PATH_TO_DATA='/content/drive/My Drive/Datasets/caliche/severstal-steel-defect-detection'
PATH_TO_PROJECT='/content/drive/My Drive/metal-sheet-defect'
file_path = PATH_TO_DATA+'/train.csv'
raw_data = pd.read_csv(file_path)
print("The number of records are : ", raw_data.shape[0])
print("The number of features are : ", raw_data.shape[1])
print("The list of features is : ", raw_data.columns)
raw_data.head()
def RLE_To_ImageMask(mask_rle, shape=(1600, 256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width, height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    # Splitting the run-length encoding
    s = mask_rle.split()
    # Creating a np array for start pixel and its length
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    # Reducing the start by 1
    starts -= 1
    # Creating a np array for end pixel
    ends = starts + lengths
    # Creating a img mask
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    # Entering 1 at the place for the defect pixels
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape).T     
def Add_Undefected_Images(data): 
    """ 
        Takes the dataframe as input and return the updated dataframe which contain both defected and undefected images information 
    """ 
    # Count the image file names for the given directory
    count = 0
    folder = PATH_TO_DATA+'/train_images'
    filename_list = []
    for filename in os.listdir(folder):
        if filename not in data["ImageId"].values:
            filename_list.append(filename)
        count += 1
    # Print the total number of images present in the directory
    print("No. of Training images provided : ", count)
    print("No. of Undefected images : ", len(filename_list))
    
    # Creating the dictionary that contain the undefected images details
    dictionary = {}
    dictionary.update({"ImageId": filename_list, "ClassId": 0, "EncodedPixels": 0, "mask_pixel_sum": 0})
    data_undefected = pd.DataFrame(dictionary)

    # Concatinate the the defected and undefected images in the single dataframe called data.
    data = pd.concat([data, data_undefected])
    
    return data
# Calculating the sum of defected pixels for the defected image
raw_data["mask_pixel_sum"] = raw_data.apply(lambda x: RLE_To_ImageMask(x["EncodedPixels"]).sum(), axis = 1)
# Adding details for the undefected images
raw_data = Add_Undefected_Images(raw_data)
raw_data["binary_class"] = raw_data["ClassId"].apply(lambda x: 0 if x == 0 else 1)
# Checking the features and no. of records in the dataset after including the details for missing images

print("The number of records are : ", raw_data.shape[0])
print("The number of features are : ", raw_data.shape[1])
print("The list of features is : ", raw_data.columns)

raw_data.tail()
print("No. of images with multiple class of defects is : ", raw_data.shape[0] - 12568)
print("The basic info about the raw data is : \n")
raw_data.info()
# Checking the missing values in the dataset
print("No. of missing values in the dataset : \n", raw_data.isnull().sum())
#================================================================================
#Exploratory Data Analysis
print('Explorataty Data Analysis')
# Plotting a countplot to visualize the distribution of reviews based on Rating Score
df_binary_clf = raw_data[["ImageId", "binary_class"]].drop_duplicates()
sns.set(style="darkgrid")
fig, axes = plt.subplots(1, 1, figsize=(25, 5))
sns.countplot(x="binary_class", data=df_binary_clf, ax=axes)
plt.title("DEFECT vs NO DEFECT", fontsize='xx-large')
plt.xlabel("CLASSES")
plt.ylabel("BINARY CLASS COUNT")
plt.show() 
#================================================================================
#Analysing the training data-Types of defects & distribution
print('Analysing the training data-Types of defects & distribution')
sns.set(style="darkgrid")
fig, axes = plt.subplots(1, 1, figsize=(25, 5))

sns.countplot(x="ClassId", data=raw_data, ax=axes)

plt.title("DISTRIBUTION OF DEFECT CLASS IN THE TRAINING DATA", fontsize='xx-large')
plt.xlabel("CLASSES")
plt.ylabel("DEFFECT CLASS COUNT")
plt.show() 
#================================================================================
#Scatter Plat for Mask Pixel Sum
print('Scatter Plat for Mask Pixel Sum')
print('Defect 1')
a = [i for i in raw_data[raw_data["ClassId"] == 1]["mask_pixel_sum"].values]
a = sorted(a)

sns.set(style="darkgrid")
fig, axes = plt.subplots(1, 1, figsize=(25, 5))

plt.title("Defect Class 1 Mask Sum Threshold Graph")
plt.scatter(a, range(len(a)))
plt.show()

print('Defect 2')
a = [i for i in raw_data[raw_data["ClassId"] == 2]["mask_pixel_sum"].values]
a = sorted(a)
sns.set(style="darkgrid")
fig, axes = plt.subplots(1, 1, figsize=(25, 5))

plt.title("Defect Class 2 Mask Sum Threshold Graph")
plt.scatter(a, range(len(a)))
plt.show()

print('Defect 3')
a = [i for i in raw_data[raw_data["ClassId"] == 3]["mask_pixel_sum"].values]
a = sorted(a)

sns.set(style="darkgrid")
fig, axes = plt.subplots(1, 1, figsize=(25, 5))

plt.title("Defect Class 3 Mask Sum Threshold Graph")
plt.scatter(a, range(len(a)))
plt.show()

print('Defect 4')
a = [i for i in raw_data[raw_data["ClassId"] == 4]["mask_pixel_sum"].values]
a = sorted(a)

sns.set(style="darkgrid")
fig, axes = plt.subplots(1, 1, figsize=(25, 5))

plt.title("Defect Class 4 Mask Sum Threshold Graph")
plt.scatter(a, range(len(a)))
plt.show()

def Show_Images(dataset, defect_type, start, end):
    """
        Showing the images of different defect type.
    """

    print(f"Defect Class {defect_type}")
    image_1 = dataset[dataset["ClassId"] == defect_type][start:end]["ImageId"].values
    for i in image_1:
        print("Image id : ", i)
        sns.set(style="darkgrid")
        plt.figure(figsize=(20, 4))
        # Loading the images one by one from the directory.
        image_read = cv2.imread(PATH_TO_DATA+'/train_images/' + i)
        image_read = cv2.cvtColor(image_read, cv2.COLOR_BGR2RGB)
        # Show the graph
        plt.imshow(image_read) 

Show_Images(raw_data, 0, 20, 24)
Show_Images(raw_data, 1, 20, 24)
Show_Images(raw_data, 2, 20, 24)
Show_Images(raw_data, 3, 20, 24)
Show_Images(raw_data, 4, 7, 11)
#===========================================================================================
#View masks
#for def1
Encoded = raw_data[raw_data["ClassId"] == 1]["EncodedPixels"].values[35]
print("Masked Image Defect Class 1")

Masked_image = RLE_To_ImageMask(Encoded)
plt.figure(figsize=(20, 4))
plt.title("Sample Image Mask for Defect Class 1", fontsize='x-large')
plt.imshow(Masked_image, cmap="viridis")
#for def2
Encoded = raw_data[raw_data["ClassId"] == 2]["EncodedPixels"].values[35]
print("Masked Image Defect Class 2")

Masked_image = RLE_To_ImageMask(Encoded)
plt.figure(figsize=(20, 4))
plt.title("Sample Image Mask for Defect Class 1", fontsize='x-large')
plt.imshow(Masked_image, cmap="viridis")
#for def3
Encoded = raw_data[raw_data["ClassId"] == 3]["EncodedPixels"].values[35]
print("Masked Image Defect Class 3")

Masked_image = RLE_To_ImageMask(Encoded)
plt.figure(figsize=(20, 4))
plt.title("Sample Image Mask for Defect Class 1", fontsize='x-large')
plt.imshow(Masked_image, cmap="viridis")
#for def4
Encoded = raw_data[raw_data["ClassId"] == 4]["EncodedPixels"].values[35]
print("Masked Image Defect Class 4")

Masked_image = RLE_To_ImageMask(Encoded)
plt.figure(figsize=(20, 4))
plt.title("Sample Image Mask for Defect Class 1", fontsize='x-large')
plt.imshow(Masked_image, cmap="viridis")