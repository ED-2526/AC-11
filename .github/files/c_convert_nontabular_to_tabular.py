import numpy as np
import os
import json
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)

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

def calcular_estadisticas(y_true, y_pred):    
    stats = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'num_muestras_test': len(y_true),
        'num_clases': len(np.unique(y_true)),
        'metricas_por_clase': classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    }
    
    return stats

def guardar_estadisticas_json(K, estadisticas, filename='estadisticas.json'):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    k_str = str(K)
    if k_str in data:
        print(f"K={K} ya existe en {filename}, saltando...")
        return False
    
    stats_serializable = {}
    for key, value in estadisticas.items():
        if isinstance(value, np.ndarray):
            stats_serializable[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            stats_serializable[key] = float(value)
        else:
            stats_serializable[key] = value

    data[k_str] = stats_serializable
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return True