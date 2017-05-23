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

########################## DECISION TREE #################################
from sklearn import tree


clf1 = tree.DecisionTreeClassifier(min_samples_split=2)
clf1 = clf1.fit(features_train, labels_train)
pred1 = clf1.predict(features_test)

clf2 = tree.DecisionTreeClassifier(min_samples_split=50)
clf2 = clf2.fit(features_train, labels_train)
pred2 = clf2.predict(features_test)

from sklearn.metrics import accuracy_score
acc1 = accuracy_score(pred1, labels_test)
acc2 = accuracy_score(pred2, labels_test)


acc_min_samples_split_2 = accuracy_score(labels_test, clf1.predict(features_test))
acc_min_samples_split_50 = accuracy_score(labels_test, clf2.predict(features_test))


def submitAccuracies():
    return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3)
    ,"acc_min_samples_split_50":round(acc_min_samples_split_50,3)}

acc = submitAccuracies()
print(acc)