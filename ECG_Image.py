# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 19:38:40 2020

@author: nr.isml
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def get_image(row_id, root= "<insert the location of the image>"):
    filename = "{}.png".format(row_id)
    file_path = os.path.join(root, filename)
    img = Image.open(file_path)
    return np.array(img)
    
def create_features(img):
    
    scale_percent = 60 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, im_gray = cv2.threshold(im_gray, 128, 192, cv2.THRESH_OTSU)
    
    #ORB FEATURE EXTRACTION
    orb = cv2.ORB_create(nfeatures=500)
    kp, des = orb.detectAndCompute(im_gray, None)
    
    #KAZE FEATURE EXTRACTION
    #kaze = cv2.KAZE_create()
    #kp, des = kaze.detectAndCompute(im_gray, None)
    
    #SIFT FEATURE EXTRACTION
    #sift = cv.SIFT_create()
    #kp, des = sift.detectAndCompute(im_gray,None)
    
    #AKAZE FEATURE EXTRACTION
    #akaze = cv2.AKAZE_create()
    #kp, des = akaze.detectAndCompute(im_gray, None)
    
    #BRISK FEATURE EXTRACTION
    #brisk = cv2.BRISK_create()
    #kp, des = brisk.detectAndCompute(im_gray, None)
    
    des = des.astype(np.float64)
    des = des[0:100,:]
    #uncomment below code if you run ORB FEATURE EXTRACTION
    vector_data = des.reshape(1,3200)
    #uncomment below code if you run KAZE OR BRISK FEATURE EXTRACTION
    #vector_data = des.reshape(1,6400)
    #uncomment below code if you run SIFT FEATURE EXTRACTION
    #vector_data = des.reshape(1,12800)
    #uncomment below code if you run AKAZE FEATURE EXTRACTION
    #vector_data = des.reshape(1,6100)
    list_data = vector_data.tolist()
    flat_features = np.hstack(list_data)
    return flat_features

def create_feature_matrix(label_dataframe):
    features_list = []
    for img_id in label_dataframe.index:
        img = get_image(img_id)
        image_features = create_features(img)
        features_list.append(image_features)
    feature_matrix = np.array(features_list)
    return feature_matrix

label = pd.read_csv("<insert the class file>", index_col=0)

feature_matrix = create_feature_matrix(label)

ss = StandardScaler()
ecg_feature = ss.fit_transform(feature_matrix)

X = ecg_feature
y = pd.Series(label.genus.values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234123)

svm = SVC(kernel='linear', gamma='auto', C=10, probability=True, random_state=42)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy is: ', accuracy)