from a_local_feature_extraction import creation_of_descriptors, resize
from b_vocabulary_extraction import split_train_test, build_vocabulary
from c_convert_nontabular_to_tabular import dataset_to_histograms, \
    calcular_estadisticas, guardar_estadisticas_json
from d_metric_visualization import visualitzar_tots_els_kernels, generar_heatmap_millors_k, analitzar_roc
from sklearn import svm
import os
import json
import numpy as np



def SVCRbf(method, K, X_train, y_train, X_test, y_test, c_flag, c, filename = 'estadisticas_rbf.json'):
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                if c_flag == True:
                    if method in data and str(c) in data[method]:
                        print(f"c={c} con método {method} ya procesada, saltant...")
                        return
                elif c_flag == False:
                    if method in data and str(K) in data[method]:
                        print(f"K={K} con método {method} ya procesada, saltant...")
                        return
        
        clf = svm.SVC(kernel='rbf', C=c)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if c_flag == True:
            guardar_estadisticas_json(method, c, calcular_estadisticas(y_test, y_pred), filename = filename)
        elif c_flag == False:
            guardar_estadisticas_json(method, K, calcular_estadisticas(y_test, y_pred), filename = filename)


def SVCSigmoid(method, K, X_train, y_train, X_test, y_test, c_flag, c, filename = 'estadisticas_sigmoide.json'):
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                if c_flag == True:
                    if method in data and str(c) in data[method]:
                        print(f"c={c} con método {method} ya procesada, saltant...")
                        return
                elif c_flag == False:
                    if method in data and str(K) in data[method]:
                        print(f"K={K} con método {method} ya procesada, saltant...")
                        return
        clf = svm.SVC(kernel='sigmoid', C=c)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if c_flag == True:
            guardar_estadisticas_json(method, c, calcular_estadisticas(y_test, y_pred), filename = filename)
        elif c_flag == False:
            guardar_estadisticas_json(method, K, calcular_estadisticas(y_test, y_pred), filename = filename)

def calcular_metricas(dim_descriptors, max_keypoints, num_clases, resized, images_for_class, methods,
                      k, c_flag = False, descriptors_flag = False, split_flag = False, vocab_flag = False):
    for method in methods:
        creation_of_descriptors(resized, [method], flag = descriptors_flag,
                                dim_descriptors = dim_descriptors,
                                max_keypoints = max_keypoints, num_classes = num_clases, images_for_class= images_for_class)

        train_data, test_data = split_train_test(split_flag, method, dim_descriptors, max_keypoints)

        for K in k:
            if c_flag == True:
                c = np.arange(0.1,5, 0.1).astype('float')
                vocabulary  = build_vocabulary(train_data, method, vocab_flag, K)
                for C in c:
                    C = round(C, 2)
                    X_train, y_train = dataset_to_histograms(train_data, vocabulary, K)
                    X_test, y_test = dataset_to_histograms(test_data, vocabulary, K)
                    SVCRbf(method, K, X_train, y_train, X_test, y_test,c_flag, C)
                    SVCSigmoid(method, K, X_train, y_train, X_test, y_test,c_flag, C)
                return
                     
            C = 1
            vocabulary  = build_vocabulary(train_data, method, vocab_flag, K)
            X_train, y_train = dataset_to_histograms(train_data, vocabulary, K)
            X_test, y_test = dataset_to_histograms(test_data, vocabulary, K)
            SVCRbf(method, K, X_train, y_train, X_test, y_test, c_flag, C)
            SVCSigmoid(method, K, X_train, y_train, X_test, y_test, c_flag, C)

def calcular_roc_curves(flag_resized, method, flag_descriptors, dim_descriptors, max_keypoints, num_classes, images_for_class,
                        flag_split, flag_kmeans, k, kernel):
    creation_of_descriptors(flag_resized, [method], flag = flag_descriptors,
                                dim_descriptors = dim_descriptors,
                                max_keypoints = max_keypoints, num_classes = num_classes, images_for_class= images_for_class)

    train_data, test_data = split_train_test(flag_split, method, dim_descriptors, max_keypoints)
    vocabulary  = build_vocabulary(train_data, method, flag_kmeans, k)
    X_train, y_train = dataset_to_histograms(train_data, vocabulary, k)
    X_test, y_test = dataset_to_histograms(test_data, vocabulary, k)
    analitzar_roc(X_train, y_train, X_test, y_test, kernel=kernel, K = k, method=method)

if __name__ == '__main__':
    lista_capetas = os.listdir(os.path.join(os.path.dirname(__file__), ".."))
    if 'Food Classification resized' not in lista_capetas:
        resize((300,300))
    num_classes = None
    methods = ['sift,grey','sift,splitted', 'harris,grey', 'harris,splitted', 'dense,grey', 'dense,splitted', 'dense_extrem,grey', 'dense_extrem,splitted' ]
    for method in methods:
        creation_of_descriptors(True,[method],True,0,0,None,0)
        split_train_test(True, method, 0,0)
    k = np.array([1,200,400,600,800,1000,1200,1300,1400,1500,1600,1700,1800,1900,2000,2200,2300,2400,2500,2600,2800,2900,3000])
    calcular_metricas(0,0,num_classes, True, 0, methods, k, c_flag = False, descriptors_flag = False, split_flag = False, vocab_flag = True)
    #calcular_roc_curves(True, 'sift,splitted', False, 0, 0, 250, num_classes, False, False, 1500, 'sigmoid')
    
