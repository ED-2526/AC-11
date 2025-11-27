import cv2
import numpy as np
import os
import pickle

def extract_sift_features(image_path):
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return descriptors

def extract_harris_sift(image_path, max_corners=0):
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    corners = cv2.goodFeaturesToTrack(gray, max_corners, 0.01, 10)
    
    if corners is None:
        return None
    
    keypoints = [cv2.KeyPoint(x=c[0][0], y=c[0][1], size=20) for c in corners]
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.compute(gray, keypoints)
    
    return descriptors

def extract_mser_sift(image_path):
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    
    keypoints = []
    for region in regions:
        if len(region) > 5:
            center = region.mean(axis=0)
            keypoints.append(cv2.KeyPoint(x=center[0], y=center[1], size=20))
    
    if len(keypoints) == 0:
        return None
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.compute(gray, keypoints)
    
    return descriptors

def extract_dense_sift(image_path, step_size=10, patch_size=16):
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    keypoints = []
    h, w = gray.shape
    for y in range(0, h - patch_size, step_size):
        for x in range(0, w - patch_size, step_size):
            keypoints.append(cv2.KeyPoint(x, y, patch_size))
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.compute(gray, keypoints)
    
    return descriptors

def extract_features_from_dataset(data_dir, method='dense'):
    methods = {
        'sift': extract_sift_features,
        'harris': extract_harris_sift,
        'mser': extract_mser_sift,
        'dense': extract_dense_sift
    }
    
    extract_fn = methods[method]
    result = {}
    
    for food_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, food_name)
        
        if not os.path.isdir(class_path):
            continue
        
        result[food_name] = {}
        print(f"\nProcesando: {food_name}")
        
        for img_name in os.listdir(class_path):
            if not img_name.endswith(('.jpg', '.png', '.jpeg')):
                continue
                
            img_path = os.path.join(class_path, img_name)
            descriptors = extract_fn(img_path)
            
            if descriptors is not None and len(descriptors) > 0:
                result[food_name][img_path] = descriptors
                print(f"  {img_name}: {len(descriptors)} features")
            else:
                print(f"  {img_name}: NO features")
    
    return result

def creation_of_descriptors(methods, flag = False):
    if flag == False:
        print("No s'ha executat la extracció perquè ja hi ha un pickle")
        return
    else:
        dir = os.path.join(os.path.dirname(__file__), '..', 'Food Classification')
        for method in methods:
            result = extract_features_from_dataset(dir, method)
            with open(os.path.join(os.path.dirname(__file__), 'features_{method}.pickle'), 'wb') as f:
                pickle.dump(result, f)

def hola():
    print("hola")