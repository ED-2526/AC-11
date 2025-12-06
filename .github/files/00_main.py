from a_local_feature_extraction import creation_of_descriptors
from b_vocabulary_extraction import vocabulary, split_train_test, build_vocabulary
from c_convert_nontabular_to_tabular import dataset_to_histograms, \
    calcular_estadisticas, guardar_estadisticas_json
from sklearn import svm
import os
import json
import numpy as np

methods = ['sift']
dim_descriptors = 0
max_keypoints = 0
num_clases = 4

########################################################################################

creation_of_descriptors(methods, flag = False,
                        dim_descriptors = dim_descriptors,
                        max_keypoints = max_keypoints, num_classes = num_clases)

train_data, test_data = split_train_test(False, 'sift', dim_descriptors, max_keypoints)
k = [500, 1000, 3000, 3500, 4000, 4500, 5000]  # Solo los que tienen pickle
for K in k:
    # Verificar si el pickle existe
    pickle_path = os.path.join(os.path.dirname(__file__), f'kmeans_sift_{K}.pickle')
    if not os.path.exists(pickle_path):
        print(f"K={K} no tiene pickle, saltant...")
        continue
    
    filepath = os.path.join(os.path.dirname(__file__), 'estadisticas_k2.json')
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
            if str(K) in data:
                print(f"K={K} ya procesada, saltant...")
                continue
    vocabulary  = build_vocabulary(train_data, methods[0], False, K)
    X_train, y_train = dataset_to_histograms(train_data, vocabulary, K)
    X_test, y_test = dataset_to_histograms(test_data, vocabulary, K)
    clf = svm.SVC(kernel='sigmoid')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    guardar_estadisticas_json(K, calcular_estadisticas(y_test, y_pred), filename= os.path.join(os.path.dirname(__file__), 'estadisticas_k2.json'))

