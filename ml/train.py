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
# f as picture file name
warnings.filterwarnings("ignore")
# image_folder = "C:\\My\\CodeLeague\\MLOps\\TrainingData\\Contest_train\\"
image_folder = "C:\\My\\CodeLeague\\Ironman\\20191217\\"
folder_name = ml.get_folder_name_in_int(image_folder)  # [3, 5, 17, 18, 19, 21, 22, 43, 45, 46, 55, 56, 59, 63, 67, 69, 71, 72, 73, 74, 77]  # folder names
folder_name

# data_X, data_Y = ml.loadImagesFromFolder(image_folder)
data_X, data_Y = ml.loadImagesAndCSVFromFolder(image_folder)


X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=23) # Split data_X and data_Y to train set and test set


# # Train a random forest classifier
# print(X_train.shape,
#     y_train.shape,
#     X_test.shape,
#     y_test.shape)

# rft_accuracy, rft_model = ml.rft_model(X_train, y_train, X_test, y_test)
# print(rft_accuracy)



best_n, best_m, max_score=ml.findBestnm(data_X,data_Y)


clf_ExT = ExtraTreesClassifier(n_estimators=best_n, max_depth=best_m, random_state=0)
scores_EXT = cross_val_score(clf_ExT, data_X, data_Y, cv = 5)
is_better = (scores_EXT.mean() > max_score)

print(scores_EXT.mean(), 'Better than the Random Forest:', is_better)



if is_better == True:
    final_model,y_pred = ml.build_final_model(ExtraTreesClassifier(n_estimators=best_n, max_depth=best_m, n_jobs=-1), X_train, y_train, X_test)
else:
    final_model,y_pred = ml.build_final_model(RandomForestClassifier(n_estimators=best_n, max_depth=best_m, n_jobs=-1), X_train, y_train, X_test)
y_pred


# # bench_data = pd.read_csv("benchmark.csv")
# # benchmark = bench_data['shot_made_flag'].values
# y_test.shape
# y_pred.shape
ml.compare_y_test_to_benchmark(y_test, y_pred)


# # Build the final model



# if is_better == True:
#     final_model = ExtraTreesClassifier(n_estimators=best_n, max_depth=best_m, n_jobs=-1)
# else:
#     final_model = RandomForestClassifier(n_estimators=best_n, max_depth=best_m, n_jobs=-1)
# final_model.fit(data_X, data_Y)


# # Try to predict one picture

# imagePath = r"./TestData/Task1 (1).jpeg"
# imagePath = "C:\\My\\CodeLeague\\MLOps\\TestData\\Task3Q1 (1).jpeg"
# print('Image {0} should be {1}'.format(imagePath, ml.FinalModelPredict(imagePath, final_model)[0]))
########################################################################################

# # Save pickle data

ml.save_model(final_model,'C:\\ironman2.pkl') 
warnings.filterwarnings("New model are generated, please load the new model")




# data_Y = np.array(data_y)
    
# print('The shape of label is: ',data_Y.shape)







# ## Handle TestData_Part1






# ## Handle TestData_Part2

