import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()

#TWEAK ORIGINAL TO COMPARE LINEARSVC & SVC
########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

clf = SVC(kernel="linear")
lin_clf = LinearSVC()
 
#### fit the classifier
#### using the training features/labels
#### make a set of predictions on the test data

clf.fit(features_train,labels_train) 
lin_clf.fit(features_train,labels_train)

#### store your predictions in a list named pred

#dec_linear = lin_clf.decision_function([[1]])
#print("dec.shape")
#print(dec.shape[1])

#dec_kernel = clf.decision_function([[1]])
#print("dec_kernel.shape[1]")
#print(dec_kernel.shape[1])

pred = clf.predict(features_test)
lin_pred = lin_clf.predict(features_test)


from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
acc_lin = accuracy_score(lin_pred,labels_test)
print("linearSVM accuracy")
print(acc_lin)
def submitAccuracy():
    return acc, acc_lin
