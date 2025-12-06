from a_local_feature_extraction import creation_of_descriptors
from b_vocabulary_extraction import vocabulary_crear, split_train_test, build_vocabulary
from c_convert_nontabular_to_tabular import dataset_to_histograms, \
    calcular_estadisticas, guardar_estadisticas_json
#from d_metric_visualization import visualitzar_tots_els_kernels, generar_heatmap_millors_k
from sklearn import svm
import os
import json
import numpy as np

dim_descriptors = 0
max_keypoints = 0
num_clases = 2


def LinearSVC(method, K, X_train, y_train, X_test, y_test, filename = 'estadisticas_LinearSVC.json'):
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                if method in data and str(K) in data[method]:
                    print(f"K={K} con método {method} ya procesada, saltant...")
                    return
        clf = svm.LinearSVC()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        guardar_estadisticas_json(method, K, calcular_estadisticas(y_test, y_pred), filename = filename)

def SVCLinear(method, K, X_train, y_train, X_test, y_test, filename = 'estadisticas_Linear.json'):
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                if method in data and str(K) in data[method]:
                    print(f"K={K} con método {method} ya procesada, saltant...")
                    return
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        guardar_estadisticas_json(method, K, calcular_estadisticas(y_test, y_pred), filename = filename)

def SVCRbf(method, K, X_train, y_train, X_test, y_test, filename = 'estadisticas_rbf.json'):
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                if method in data and str(K) in data[method]:
                    print(f"K={K} con método {method} ya procesada, saltant...")
                    return
        clf = svm.SVC(kernel='rbf')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        guardar_estadisticas_json(method, K, calcular_estadisticas(y_test, y_pred), filename = filename)

def SVCSigmoid(method, K, X_train, y_train, X_test, y_test, filename = 'estadisticas_sigmoide.json'):
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                if method in data and str(K) in data[method]:
                    print(f"K={K} con método {method} ya procesada, saltant...")
                    return
        clf = svm.SVC(kernel='sigmoid')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        guardar_estadisticas_json(method, K, calcular_estadisticas(y_test, y_pred), filename = filename)

def SVCPoly3(method, K, X_train, y_train, X_test, y_test, filename = 'estadisticas_poly3.json'):
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                if method in data and str(K) in data[method]:
                    print(f"K={K} con método {method} ya procesada, saltant...")
                    return
        clf = svm.SVC(kernel='poly')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        guardar_estadisticas_json(method, K, calcular_estadisticas(y_test, y_pred), filename = filename)

def SVCPoly5(method, K, X_train, y_train, X_test, y_test, filename = 'estadisticas_poly5.json'):
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                if method in data and str(K) in data[method]:
                    print(f"K={K} con método {method} ya procesada, saltant...")
                    return
        clf = svm.SVC(kernel='poly', degree=5)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        guardar_estadisticas_json(method, K, calcular_estadisticas(y_test, y_pred), filename = filename)



########################################################################################
methods = ['dense,grey', 'harris,splitted']
k = np.array([10,25,50,100,200,250,500,1000,2000,3000,5000,6000,7000,8000,9000,10000])
for method in methods:
    creation_of_descriptors([method], flag = True,
                            dim_descriptors = dim_descriptors,
                            max_keypoints = max_keypoints, num_classes = num_clases)

    train_data, test_data = split_train_test(True, method, dim_descriptors, max_keypoints)

    for K in k:
        vocabulary  = build_vocabulary(train_data, method, True, K)
        X_train, y_train = dataset_to_histograms(train_data, vocabulary, K)
        X_test, y_test = dataset_to_histograms(test_data, vocabulary, K)
        LinearSVC(method, K, X_train, y_train, X_test, y_test)
        SVCLinear(method, K, X_train, y_train, X_test, y_test)
        SVCRbf(method, K, X_train, y_train, X_test, y_test)
        SVCSigmoid(method, K, X_train, y_train, X_test, y_test)
        SVCPoly3(method, K, X_train, y_train, X_test, y_test)
        SVCPoly5(method, K, X_train, y_train, X_test, y_test)

#visualitzar_tots_els_kernels()
#generar_heatmap_millors_k()