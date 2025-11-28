from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pickle
import os
import numpy as np

def split_train_test(flag, method, test_size=0.2, random_state=42):
    if flag == False:
        print("Split ja fet, carregant desde pickle...")
        with open(os.path.join(os.path.dirname(__file__), f'train_data_{method}.pickle'), 'rb') as f:
            train_dict = pickle.load(f)
        with open(os.path.join(os.path.dirname(__file__), f'test_data_{method}.pickle'), 'rb') as f:
            test_dict = pickle.load(f)
    else:       
        with open(os.path.join(os.path.dirname(__file__), f'features_{method}.pickle'), 'rb') as f:
            features_dict = pickle.load(f)

        train_dict = {}
        test_dict = {}
        
        for food_name, images_dict in features_dict.items():
            paths = list(images_dict.keys())
            
            train_paths, test_paths = train_test_split(
                paths, 
                test_size=test_size, 
                random_state=random_state
            )
            
            train_dict[food_name] = {p: images_dict[p] for p in train_paths}
            test_dict[food_name] = {p: images_dict[p] for p in test_paths}
            
        with open(os.path.join(os.path.dirname(__file__), f'train_data_{method}.pickle'), 'wb') as f:
            pickle.dump(train_dict, f)
        with open(os.path.join(os.path.dirname(__file__), f'test_data_{method}.pickle'), 'wb') as f:
            pickle.dump(test_dict, f)

    for key in train_dict.keys():
        print(f"{key}: {len(train_dict[key])} train, {len(test_dict[key])} test")
    
    return train_dict, test_dict


def build_vocabulary(train_dict, method, flag, K):
    if flag == False:
        print("Carregant el kmenas des del pickle")
        with open(os.path.join(os.path.dirname(__file__), f'kmeans_{method}_{K}.pickle'), 'rb') as f:
            kmeans = pickle.load(f)
    else:
        all_descriptors = []
        
        for food_name, images_dict in train_dict.items():
            for img_path, descriptors in images_dict.items():
                all_descriptors.append(descriptors)
        
        all_descriptors_numpy = np.vstack(all_descriptors)
        print(f"Total descriptors: {all_descriptors_numpy.shape}")
        
        print(f"Entrenando K-Means con K={K}...")
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10, verbose= True)
        kmeans.fit(all_descriptors_numpy)
        with open(os.path.join(os.path.dirname(__file__), f'kmeans_{method}_{K}.pickle'), 'wb') as f:
            pickle.dump(kmeans, f)

        
        print(f"Vocabulario creado: {kmeans.cluster_centers_.shape}")
        
    return kmeans

def vocabulary(flag_split, flag_kmeans, k, method = 'sift'):
    train_data, test_data = split_train_test(flag_split, method)
    kmeans = build_vocabulary(train_data, method, flag_kmeans, k)
    return train_data, test_data, kmeans
