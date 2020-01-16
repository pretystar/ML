import matplotlib.pyplot as plt
import time
import path
import numpy as np
import pandas as pd
from skimage import io,data,color,exposure,transform
from sklearn import preprocessing, svm
import sklearn
# sklearn.__version__
import os, fnmatch
import warnings
import ml

from skimage.color import rgb2gray, gray2rgb
from skimage.transform import rescale, resize, downscale_local_mean

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, make_scorer, f1_score, accuracy_score, recall_score, precision_score, log_loss

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm

import pickle

def findBestnm(data_X,data_Y):
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
    print(best_n, best_m)
    return best_n, best_m, max_score
def loadImagesAndCSVFromFolder(image_folder):
    data_x = []
    data_y = []
    folder_name = get_folder_name_in_int(image_folder)
    print('Image folders are:')
    
    for i in range(len(folder_name)):
        print(folder_name[i])
        pic_folder=image_folder+ str(folder_name[i])+'\\'
        print(pic_folder)
        pic_folder  = os.path.normpath(pic_folder)
        # result_file_names = fnmatch.filter(os.listdir(pic_folder), 'result.txt')
        resultPath = os.path.normpath(pic_folder+"\\answerResult.txt")
        try:
            with open(resultPath,'r') as f:
                print(f.readlines())
                # Do something with the file
        except IOError:
            print("File not accessible")
            continue
        # if os.path.exists((resultPath)):
        #     print ("result File exist")
        # else:
        #     print ("result File not exist")
        #     continue
        with open(resultPath, newline='') as text_file:
            content = text_file.readlines()
            # data_y.append(list(map(int, str.split(content[0],","))))
            data_y += list(map(int, str.split(content[0],",")))

        file_names = fnmatch.filter(os.listdir(pic_folder+"\\img\\"), '*.jpeg')
        print(pic_folder)
        for j in range(len(file_names)):
            img = ml.batch_process(pic_folder+"\\img\\"+file_names[j])
            data_x.append(img)
            # data_y.append(folder_name[i])
    data_X = np.array(data_x)
    data_Y = np.array(data_y)
    print('The shape of feature set (set of image arrays) is: ',data_X.shape)
    print('The shape of label is: ',data_Y.shape)
    return data_X,data_Y

def loadImagesFromFolder(image_folder):
    data_x = []
    data_y = []
    folder_name = get_folder_name_in_int(image_folder)
    print('Image folders are:')
    for i in range(len(folder_name)):
        pic_folder=image_folder+ '/'+ str(folder_name[i])+'/'
        file_names = fnmatch.filter(os.listdir(pic_folder), '*.jpeg')
        print(pic_folder)
        for j in range(len(file_names)):
            img = ml.batch_process(pic_folder+file_names[j])
            data_x.append(img)
            data_y.append(folder_name[i])
    data_X = np.array(data_x)
    data_Y = np.array(data_y)
    print('The shape of feature set (set of image arrays) is: ',data_X.shape)
    print('The shape of label is: ',data_Y.shape)
    return data_X,data_Y
# def loadImageFromSingleFolder(folder):
#     data_x = []
#     data_y = []
    
def FinalModelPredict(imgPath, final_model):
    X = []
    sk_image = batch_process(imgPath)
    # Reshape sk_image
    X.append(sk_image)
#     sk_image = sk_image.reshape(1,625)
    print(sk_image)
    result = final_model.predict(X)
    return result

def save_model(model, path):
  f = open(path,'wb')
  pickle.dump(model, f)

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

def predict_and_save_data_to_file(final_model, data, out_file_name):
    y_predict = final_model.predict(data)
    y_predict_df = pd.DataFrame(y_predict)
    print(y_predict_df.shape)
    print(y_predict_df.head())
    print('y_predict are saved in file {0}'.format(out_file_name))
    y_predict_df.to_csv(out_file_name, header=False, index=False, encoding='ansi')
    return y_predict_df

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

def build_final_model(model, X_train, y_train, X_test):
#     t1 = time.time()

    # model = RandomForestClassifier(n_estimators=best_n, max_depth=best_m, n_jobs=-1)
#     model = RandomForestClassifier(n_estimators=best_n, max_depth=best_m, n_jobs=-1)
    
    model.fit(X_train, y_train)

    # t2 = time.time()
#     print('Done processing {0} trees ({1:.3f}sec)'.format(best_n, t2-t1))

    pred_result = model.predict(X_test)
    return model,pred_result

def get_folder_name_in_int(image_folder):
    folder_names = os.listdir(image_folder)
    folder_names_numeric = []
    for folder_name in folder_names:
        if(os.path.isdir(image_folder+"\\"+folder_name)):
            folder_names_numeric.append(folder_name)
    folder_names_numeric.sort()
    return folder_names_numeric

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

def pic_normalization(pic):
    pic = pic-pic.min()
    pic = pic/pic.max()
    # Recover range to [0,255] and return
    return pic*255

def batch_process(pic_path):
    sk_image = pic_preprocess(pic_path)
    sk_image = pic_normalization(sk_image).astype(np.uint8)
    # Reshape sk_image from matrix to 1-dimensional array
    image_array = sk_image.flatten()
    
    return image_array


def rft_model(X_train, y_train, X_valid, y_valid):
    train_model = RandomForestClassifier(random_state=0) # Create a Random Forest Classifier
    train_model.fit(X_train, y_train) # Train RandForestClassifier with X_train and y_train
    #print(X_valid)
    y_predict = train_model.predict(X_valid)# Predict using X_valid
    #print(y_predict)
    rft_result_accuracy = accuracy_score(y_valid, y_predict) # Compute accuracy
    return rft_result_accuracy, train_model

