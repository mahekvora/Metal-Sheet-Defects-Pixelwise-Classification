#Data Cleaning and Feature engineering
import numpy as np
import pandas as pd
import os

PATH_TO_DATA='/content/drive/My Drive/Datasets/caliche/severstal-steel-defect-detection'
PATH_TO_PROJECT='/content/drive/My Drive/metal-sheet-defect'
file_path = PATH_TO_DATA+'/train.csv'
raw_data = pd.read_csv(file_path)

#Data Cleaning
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
    folder = "/content/drive/My Drive/Datasets/caliche/severstal-steel-defect-detection/train_images"
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

print("No. of images with multiple class of defects is : ", raw_data.shape[0] - 12568)
print("The basic info about the raw data is : \n")
raw_data.info()
# Checking the missing values in the dataset
print("No. of missing values in the dataset : \n", raw_data.isnull().sum())
# Creating a new feature by combining the image name and class
raw_data["ImageId_ClassId"] = raw_data["ImageId"] + "_" + raw_data["ClassId"].astype(str) 
# Sorting the df based on image name and class
raw_data = raw_data.sort_values(by=["ImageId_ClassId"], axis=0, ignore_index=True)

# Grouping the ImageId together to extract IDs with multiple defect classes
img_class_data = raw_data.groupby(["ImageId"])["ClassId"].agg(["unique"]).reset_index()

# Renaming the feature to "Class"
img_class_data.rename(columns = {"unique": "Class"}, inplace = True)

# Adding feature for each class
img_class_data["Class_0"] = 0
img_class_data["Class_1"] = 0
img_class_data["Class_2"] = 0
img_class_data["Class_3"] = 0
img_class_data["Class_4"] = 0

# Boolean feature indicating if defect exists
img_class_data["Any_Class"] = 0

# Entering 1 in the respective Class feature and binary "Any_Class" feature

for idx, row in img_class_data.iterrows():
    if 0 in row.Class:
        img_class_data.at[idx, 'Class_0'] = 1
    if 1 in row.Class:
        img_class_data.at[idx, 'Class_1'] = 1
        img_class_data.at[idx, 'Any_Class'] = 1
    if 2 in row.Class:
        img_class_data.at[idx, 'Class_2'] = 1
        img_class_data.at[idx, 'Any_Class'] = 1
    if 3 in row.Class:
        img_class_data.at[idx, 'Class_3'] = 1
        img_class_data.at[idx, 'Any_Class'] = 1
    if 4 in row.Class:
        img_class_data.at[idx, 'Class_4'] = 1
        img_class_data.at[idx, 'Any_Class'] = 1

img_class_data["Sum_of_Defects"] = img_class_data["Class_1"] + img_class_data["Class_2"] + img_class_data["Class_3"] + img_class_data["Class_4"]


# basic informations regarding classes

print("no. of unique Images is : ", len(img_class_data))

print("no. of Images with defects is : ", len(img_class_data[img_class_data["Any_Class"] == 1]))
print("no. of Images without defects is : ", len(img_class_data[img_class_data["Any_Class"] == 0]))

print("no. of Images with Class Defect 1 is : ", len(img_class_data[(img_class_data["Class_1"] == 1) & (img_class_data["Sum_of_Defects"] == 1)]))
print("no. of Images with Class Defect 2 is : ", len(img_class_data[(img_class_data["Class_2"] == 1) & (img_class_data["Sum_of_Defects"] == 1)]))
print("no. of Images with Class Defect 3 is : ", len(img_class_data[(img_class_data["Class_3"] == 1) & (img_class_data["Sum_of_Defects"] == 1)]))
print("no. of Images with Class Defect 4 is : ", len(img_class_data[(img_class_data["Class_4"] == 1) & (img_class_data["Sum_of_Defects"] == 1)]))

print("no. of Images with 1 Class Defects is : ", len(img_class_data[img_class_data["Sum_of_Defects"] == 1]))
print("no. of Images with 2 Class Defects is : ", len(img_class_data[img_class_data["Sum_of_Defects"] == 2]))
print("no. of Images with 3 Class Defects is : ", len(img_class_data[img_class_data["Sum_of_Defects"] == 3]))
print("no. of Images with 4 Class Defects is : ", len(img_class_data[img_class_data["Sum_of_Defects"] == 4]))