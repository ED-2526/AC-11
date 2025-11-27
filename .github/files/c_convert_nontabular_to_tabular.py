import numpy as np

def image_to_histogram(descriptors, kmeans, K):
    labels = kmeans.predict(descriptors)
    histogram, _ = np.histogram(labels, bins=range(K+1))
    histogram = histogram / histogram.sum()
    return histogram

def dataset_to_histograms(data_dict, kmeans, K):
    X = []
    y = []
    
    for food_name, images_dict in data_dict.items():
        for img_path, descriptors in images_dict.items():
            hist = image_to_histogram(descriptors, kmeans, K)
            X.append(hist)
            y.append(food_name)
    
    return np.array(X), np.array(y)