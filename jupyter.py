#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[ ]:


import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from skimage import io,data,color,exposure,transform

get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


from sklearn import preprocessing, svm


# In[6]:


import sklearn
sklearn.__version__


# # Load Image

# In[10]:


import os

def get_folder_name_in_int(image_folder):
    folder_names = os.listdir(image_folder)
    folder_names_numeric = []
    for folder_name in folder_names:
        folder_names_numeric.append(int(folder_name))    
    folder_names_numeric.sort()
    return folder_names_numeric


# In[11]:


import warnings
warnings.filterwarnings("ignore")
image_folder = r'./Contest_train_After'
folder_name = get_folder_name_in_int(image_folder)  # [3, 5, 17, 18, 19, 21, 22, 43, 45, 46, 55, 56, 59, 63, 67, 69, 71, 72, 73, 74, 77]  # folder names
folder_name


# Read one picture as skimage and plot

# In[12]:


sk_image = io.imread(image_folder+'/5/img35.jpeg')
io.imshow(sk_image)


# # Preprocess pictures

# In[13]:


from skimage.color import rgb2gray, gray2rgb
from skimage.transform import rescale, resize, downscale_local_mean
# f as picture file name
def pic_preprocess(f): 
    processed_image = []
    # Read the picture as skimage
    sk_image = io.imread(f)
    # Convert the skimage from RGB to gray scale
    img_gray = rgb2gray(sk_image)
    # Resize the picture to a fixed size of pixels
    processed_image = resize(img_gray, (25, 25), anti_aliasing = True)
    ##processed_image = resize(sk_image, (200, 200))
    # Return preprocessed picture
    return processed_image


# In[14]:


# Normalize the range of grayscale to [0,1] to maximize the contrast
def pic_normalization(pic):
    pic = pic-pic.min()
    pic = pic/pic.max()
    # Recover range to [0,255] and return
    return pic*255


# ## Picture Preprocessing Result

# In[16]:


sk_image = io.imread(image_folder+'/5/img35.jpeg')
plt.figure("hist",figsize=(12,12))
plt.subplot(321)
io.imshow(sk_image)
plt.title('original')

plt.subplot(322)
sk_image2 = pic_preprocess(image_folder+'/5/img35.jpeg')
io.imshow(sk_image2)
plt.title('Gray scaled and resized')

plt.subplot(323)
sk_image3 = pic_normalization(sk_image2)
io.imshow(sk_image3)
plt.title('Normalized')


# ## Prepare Training data

# In[17]:


### The purpose of this function is to read, preprocess, normalize and reshape a picture ## 
def batch_process(pic_path):
    sk_image = pic_preprocess(pic_path)
    sk_image = pic_normalization(sk_image).astype(np.uint8)
    # Reshape sk_image from matrix to 1-dimensional array
    image_array = sk_image.flatten()
    
    return image_array


# In[18]:


import os, fnmatch
data_x = []
data_y = []
print('Image folders are:')
for i in range(len(folder_name)):
    pic_folder=image_folder+ '/'+ str(folder_name[i])+'/'
    file_names = fnmatch.filter(os.listdir(pic_folder), '*.jpeg')
    print(pic_folder)
    t = 0
    for j in range(len(file_names)):
        img = batch_process(pic_folder+file_names[j])
        data_x.append(img)
        data_y.append(folder_name[i])
data_X = np.array(data_x)
data_Y = np.array(data_y)
print('The shape of feature set (set of image arrays) is: ',data_X.shape)
print('The shape of label is: ',data_Y.shape)


# In[28]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=23) # Split data_X and data_Y to train set and test set


# # Train a random forest classifier

# In[29]:


print(X_train.shape,
    y_train.shape,
    X_test.shape,
    y_test.shape)


# In[30]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, make_scorer, f1_score, accuracy_score, recall_score, precision_score, log_loss


def rft_model(X_train, y_train, X_valid, y_valid):
    train_model = RandomForestClassifier(random_state=0) # Create a Random Forest Classifier
    train_model.fit(X_train, y_train) # Train RandForestClassifier with X_train and y_train
    #print(X_valid)
    y_predict = train_model.predict(X_valid)# Predict using X_valid
    #print(y_predict)
    rft_result_accuracy = accuracy_score(y_valid, y_predict) # Compute accuracy
    return rft_result_accuracy, train_model

rft_accuracy, rft_model = rft_model(X_train, y_train, X_test, y_test)
print(rft_accuracy)


# In[31]:


# find the best n_estimators for RandomForestClassifier
print('Finding best n_estimators for RandomForestClassifier...')
max_score = -1
best_n = 0
scores_n = []
range_n = [1, 10, 100, 200]  # np.logspace(0,2,num=3).astype(int)
for n in range_n:
    print("the number of trees : {0}".format(n))
    t1 = time.time()
    
    rfc_score = 0.
    rfc = RandomForestClassifier(n_estimators=n, oob_score=False, random_state=10, n_jobs=-1)
    kf = KFold(n_splits=10, shuffle=True)
    for train_k, test_k in kf.split(data_X):
        rfc.fit(pd.DataFrame(data_X).iloc[train_k], pd.DataFrame(data_Y).iloc[train_k])
        #rfc_score += rfc.score(train.iloc[test_k], train_y.iloc[test_k])/10
        pred = rfc.predict(data_X[test_k])
        rfc_score += accuracy_score(data_Y[test_k], pred) / 10
    scores_n.append(rfc_score)
    if rfc_score > max_score:
        max_score = rfc_score
        best_n = n
        
    t2 = time.time()
    print('Done processing {0} trees ({1:.3f}sec)'.format(n, t2-t1))
print('best_n_estimators={0}, max_score={1}'.format(best_n, max_score))


# In[32]:


# find best max_depth for RandomForestClassifier
print('Finding best max_depth for RandomForestClassifier...')
max_score = -1
best_m = 0
scores_m = []
range_m = [1, 10, 100]  # np.logspace(0,2,num=3).astype(int)
for m in range_m:
    print("the max depth : {0}".format(m))
    t1 = time.time()
    
    rfc_score = 0.
    rfc = RandomForestClassifier(max_depth=m, n_estimators=best_n, n_jobs=-1)
    kf = KFold(n_splits=10, shuffle=True)
    for train_k, test_k in kf.split(data_X):
        rfc.fit(data_X[train_k], data_Y[train_k])
        #rfc_score += rfc.score(train.iloc[test_k], train_y.iloc[test_k])/10
        pred = rfc.predict(data_X[test_k])
        rfc_score += accuracy_score(data_Y[test_k], pred) / 10
    scores_m.append(rfc_score)
    if rfc_score > max_score:
        max_score = rfc_score
        best_m = m
    
    t2 = time.time()
    print('Done processing {0} trees ({1:.3f}sec)'.format(m, t2-t1))
print('best_max_depth={0}, max_score={1}'.format(best_m, max_score))


# Show accuracy of the Random Forest model in the validation set

# In[33]:


print(best_n, best_m)


# In[34]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm

clf_ExT = ExtraTreesClassifier(n_estimators=best_n, max_depth=best_m, random_state=0)
scores_EXT = cross_val_score(clf_ExT, data_X, data_Y, cv = 5)
is_better = (scores_EXT.mean() > max_score)

print(scores_EXT.mean(), 'Better than the Random Forest:', is_better)


# In[35]:


def build_final_model(model, X_train, y_train, X_test):
#     t1 = time.time()

    # model = RandomForestClassifier(n_estimators=best_n, max_depth=best_m, n_jobs=-1)
#     model = RandomForestClassifier(n_estimators=best_n, max_depth=best_m, n_jobs=-1)
    
    model.fit(X_train, y_train)

    t2 = time.time()
#     print('Done processing {0} trees ({1:.3f}sec)'.format(best_n, t2-t1))

    pred_result = model.predict(X_test)
    return pred_result


# In[36]:


if is_better == True:
    y_pred = build_final_model(ExtraTreesClassifier(n_estimators=best_n, max_depth=best_m, n_jobs=-1), X_train, y_train, X_test)
else:
    y_pred = build_final_model(RandomForestClassifier(n_estimators=best_n, max_depth=best_m, n_jobs=-1), X_train, y_train, X_test)
y_pred


# In[37]:


def compare_y_test_to_benchmark(benchmark, y_test):
    start_idx = 100
    stop_idx = 200
    x = range(stop_idx-start_idx)
    y1 = y_test[start_idx:stop_idx]
    y2 = benchmark[start_idx:stop_idx]
    
    plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(x,y2, "b", label='benchmark')
    plt.plot(x,y1, "g", label='predict')
    plt.legend()
    plt.show()
    
    plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(benchmark, y_test, 'bo', label='Predicted')
    plt.plot(benchmark, benchmark, 'g', label='Line')
    plt.ylabel('shot made probability')
    plt.legend()
    plt.show()

#     print("Logloss is", logloss(benchmark, y_test)) # The smaller the better
    print("accuracy_score is ", accuracy_score(benchmark, y_test))


# In[38]:


# bench_data = pd.read_csv("benchmark.csv")
# benchmark = bench_data['shot_made_flag'].values
y_test.shape
y_pred.shape
compare_y_test_to_benchmark(y_test, y_pred)


# # Build the final model

# In[39]:


if is_better == True:
    final_model = ExtraTreesClassifier(n_estimators=best_n, max_depth=best_m, n_jobs=-1)
else:
    final_model = RandomForestClassifier(n_estimators=best_n, max_depth=best_m, n_jobs=-1)
final_model.fit(data_X, data_Y)


# # Try to predict one picture

# In[40]:


def FinalModelPredict(imgPath, final_model):
    X = []
    sk_image = batch_process(imgPath)
    # Reshape sk_image
    X.append(sk_image)
#     sk_image = sk_image.reshape(1,625)
    print(sk_image)
    result = final_model.predict(X)
    return result


# In[41]:


imagePath = r"./TestData/Task1 (1).jpeg"
print('Image {0} should be {1}'.format(imagePath, FinalModelPredict(imagePath, final_model)[0]))


# # Save pickle data

# In[42]:


import pickle

def save_model(model, path):
  f = open(path,'wb')
  pickle.dump(model, f)

save_model(final_model,'irnman.pkl') 


# In[ ]:


def get_test_data(folder):
    X_test = []
    print('Image folders are:{0}'.format(folder))

    file_names = fnmatch.filter(os.listdir(folder), '*.jpeg')
#     coll.sort(key=lambda x:int(x[:-5]))
    file_names.sort(key=lambda x:len(x))  # take care of folder sequence, 1.jpeg, 2.jpeg, 3.jpeg
    print(file_names)
    
    for j in range(len(file_names)):
        img = batch_process(folder+file_names[j])
        X_test.append(img)

    X_test = np.array(X_test)
    print('The shape of feature set (set of image arrays) is: ',X_test.shape)
    return X_test
# data_Y = np.array(data_y)
    
# print('The shape of label is: ',data_Y.shape)


# In[ ]:


def predict_and_save_data_to_file(final_model, data, out_file_name):
    y_predict = final_model.predict(data)
    y_predict_df = pd.DataFrame(y_predict)
    print(y_predict_df.shape)
    print(y_predict_df.head())
    print('y_predict are saved in file {0}'.format(out_file_name))
    y_predict_df.to_csv(out_file_name, header=False, index=False, encoding='ansi')
    return y_predict_df


# ## Handle TestData_Part1

# In[ ]:


# read and predict test_data_part1
test_image_folder1 = r'./TestData_Part1/Contest_test/'
X_test_part1 = get_test_data(test_image_folder1)
X_test_part1.shape
y_predict1 = predict_and_save_data_to_file(final_model, X_test_part1, 'Ironman_part_1.csv')


# ## Handle TestData_Part2

# In[ ]:


# rename jpg to jpeg
import os
test_image_folder2 = r'./TestData_Part2/test/'
file_names = fnmatch.filter(os.listdir(test_image_folder2), '*.jpg')
for item_jpg in file_names:
    os.rename(test_image_folder2+item_jpg, test_image_folder2+item_jpg[:-3]+'jpeg')


# In[ ]:


# read and predict test_data_part2
X_test_part2 = get_test_data(test_image_folder2)
X_test_part2.shape
y_predict2 = predict_and_save_data_to_file(final_model, X_test_part2, 'Ironman_part_2.csv')

