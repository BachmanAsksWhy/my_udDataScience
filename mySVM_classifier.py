import sys
from time import time
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here





#########################################################
### your code goes here ###
def SVMAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.svm import SVC

    ### create classifier
    clf = SVC(kernel="rbf", gamma=1000, C=1)
    t0 = time()
    ### fit the classifier on the training features and labels
    clf.fit(features_train, labels_train)
    print ("training time:", round(time()-t0, 3), "s")

    ### use the trained classifier to predict labels for the test features
    t1 = time()
    pred = clf.predict(features_test)
    print ("predict time:", round(time()-t1, 3), "s")

    prettyPicture(clf, features_test, labels_test)
    output_image("test.png", "png", open("test.png", "rb").read())


    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example, 
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(pred, labels_test)
    return accuracy

def submitAccuracy():
    accuracy = SVMAccuracy(features_train, labels_train, features_test, labels_test)
    return accuracy

acc = submitAccuracy()
print(acc)

# ### draw the decision boundary with the text points overlaid
# prettyPicture(clf, features_test, labels_test)
# output_image("test.png", "png", open("test.png", "rb").read())