
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

import pickle


# # read and predict test_data_part1
# test_image_folder1 = r'./TestData/'
# X_test_part1 = ml.get_test_data(test_image_folder1)
# X_test_part1.shape
# y_predict1 = ml.predict_and_save_data_to_file(final_model, X_test_part1, 'Ironman_part_1.csv')

def predictFolder(test_image_folder,csvfile):
    # rename jpg to jpeg
    # test_image_folder2 = r'./TestData_Part2/test/'
    with open("C:\\ironman1.pkl", 'rb') as f:
        final_model = pickle.load(f)
    # f=open('C:\\ironman.pkl','r')  
    # final_model=pickle.load(f)  


    # file_names = fnmatch.filter(os.listdir(test_image_folder), '*.jpg')
    # for item_jpg in file_names:
    #     os.rename(test_image_folder+item_jpg, test_image_folder+item_jpg[:-3]+'jpeg')


    # read and predict test_data_part2
    X_test_part2 = ml.get_test_data(test_image_folder)
    # X_test_part2.shape
    y_predict2 = ml.predict_and_save_data_to_file(final_model, X_test_part2, csvfile)

while True:
    print("New loop")
    time.sleep(5)
    image_folder = "C:\\My\\CodeLeague\\Ironman\\20191217\\"
    folder_names=ml.get_folder_name_in_int(image_folder)
    for folder_name in folder_names:
        # test_image_folder2 = "C:\\My\\CodeLeague\\code\\TestData\\"
        rootfolder = image_folder + str(folder_name) +"\\img\\"
        csvfile=rootfolder + "\\predict1.csv"
        try:
            with open(csvfile,'r') as f:
                print(f.readlines())
                continue
        except IOError:
            print("File not accessible")
        predictFolder(rootfolder,csvfile)
##if exits predicted.text skip, else predict and generate csvfile 
# test_image_folder2 = "C:\\My\\CodeLeague\\code\\TestData\\"
# csvfile='Ironman_part_2.csv'
# predictFolder(test_image_folder2,csvfile)
