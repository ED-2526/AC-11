from a_local_feature_extraction import creation_of_descriptors
from b_vocabulary_extraction import vocabulary, split_train_test, build_vocabulary
from c_convert_nontabular_to_tabular import dataset_to_histograms, calcular_estadisticas
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pickle
import os

methods = ['sift']
dim_descriptors = 0
max_keypoints = 0
num_clases = 4

estadisticas = {}
creation_of_descriptors(methods, flag = True,
                        dim_descriptors = dim_descriptors,
                        max_keypoints = max_keypoints, num_classes = num_clases)
train_data, test_data = split_train_test(True, 'sift', dim_descriptors, max_keypoints)
num_vocabulari = [10, 50, 100, 200, 500, 1000, 2000, 3000, 5000]
for K in num_vocabulari:
    vocabulary  = build_vocabulary(train_data, methods[0], True, K)
    X_train, y_train = dataset_to_histograms(train_data, vocabulary, K)
    X_test, y_test = dataset_to_histograms(test_data, vocabulary, K)
    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    estadisticas[K] = calcular_estadisticas(y_test, y_pred)

with open(os.path.join(os.path.dirname(__file__), 'estadisticas.pickle'), 'wb') as f:
    pickle.dump(estadisticas, f)


