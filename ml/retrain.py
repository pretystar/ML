import matplotlib.pyplot as plt
import time
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
import csv
import pickle

def retrain(pic_folder,csvfile):
    # rename jpg to jpeg
    # test_image_folder2 = r'./TestData_Part2/test/'
    data_x = []
    data_y = []
    with open("C:\\ironman.pkl", 'rb') as f:
        final_model = pickle.load(f)
    # pic_folder=image_folder+ '/'+ str(folder_name[i])+'/'
    file_names = fnmatch.filter(os.listdir(pic_folder), '*.jpeg')
    print(pic_folder)
    for j in range(len(file_names)):
        img = ml.batch_process(pic_folder+file_names[j])
        data_x.append(img)
        # data_y.append(folder_name[i])
    file_names = fnmatch.filter(os.listdir(pic_folder), 'result.txt')
    with open(file_names[0], newline='') as text:
        data_y = text_file.readlines()

    data_X = np.array(data_x)
    data_Y = np.array(data_y)
    final_model.fit(data_X, data_Y)

    ml.save_model(final_model,'C:\\ironman1.pkl') 