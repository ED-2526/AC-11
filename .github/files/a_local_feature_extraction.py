from sklearn.decomposition import PCA
import cv2
import numpy as np
import os
import pickle

def extract_sift_features(image_path, max_keypoints):
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=max_keypoints if max_keypoints else 0)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return descriptors

def extract_harris_sift(image_path, max_keypoints):
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    max_corners = max_keypoints if max_keypoints else 0
    corners = cv2.goodFeaturesToTrack(gray, max_corners, 0.01, 10)
    
    if corners is None:
        return None
    
    keypoints = [cv2.KeyPoint(x=c[0][0], y=c[0][1], size=20) for c in corners]
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.compute(gray, keypoints)
    
    return descriptors

def extract_mser_sift(image_path, max_keypoints):
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

    if max_keypoints and len(keypoints) > max_keypoints:
        import random
        random.shuffle(keypoints)
        keypoints = keypoints[:max_keypoints]
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.compute(gray, keypoints)
    
    return descriptors

def extract_dense_sift(image_path,  max_keypoints, step_size=10, patch_size=16):
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints = []
    h, w = gray.shape
    for y in range(0, h - patch_size, step_size):
        for x in range(0, w - patch_size, step_size):
            keypoints.append(cv2.KeyPoint(x, y, patch_size))

    if max_keypoints and len(keypoints) > max_keypoints:
        indices = np.linspace(0, len(keypoints)-1, max_keypoints, dtype=int)
        keypoints = [keypoints[i] for i in indices]

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.compute(gray, keypoints)
    
    return descriptors

def extract_features_from_dataset(data_dir, max_keypoints, dim_descriptors, method='dense'):
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
            descriptors = extract_fn(img_path, max_keypoints)
            
            if descriptors is not None and len(descriptors) > 0:
                result[food_name][img_path] = descriptors
                print(f"  {img_name}: {len(descriptors)} features")
            else:
                print(f"  {img_name}: NO features")

    all_descriptors = []
    for food_name in result:
        for img_path in result[food_name]:
            descriptors = result[food_name][img_path]
            all_descriptors.append(descriptors)
    
    all_descriptors = np.vstack(all_descriptors)
    print(f"\nTotal de descriptores recolectados: {all_descriptors.shape[0]}")
    print(f"Dimensionalidad original: {all_descriptors.shape[1]}D")
    
    pca = PCA(n_components=dim_descriptors)
    pca.fit(all_descriptors)
    
    varianza_explicada = pca.explained_variance_ratio_.sum()
    for food_name in result:
        print(f"\nTransformando: {food_name}")
        for img_path in result[food_name]:
            descriptors_original = result[food_name][img_path]
            descriptors_reduced = pca.transform(descriptors_original)
            result[food_name][img_path] = descriptors_reduced
    
    return result

def creation_of_descriptors(methods, flag = False, dim_descriptors = 64, max_keypoints = 500):
    if flag == False:
        print("No s'ha executat la extracció perquè no es vol" \
        "sobreescriure lo queja hi ha un pickle")
        return
    else:
        dir = os.path.join(os.path.dirname(__file__), '..', 'Food Classification')
        for method in methods:
            result = extract_features_from_dataset(dir, max_keypoints,  dim_descriptors, method = method)
            with open(os.path.join(os.path.dirname(__file__), f'features_{method}.pickle'), 'wb') as f:
                pickle.dump(result, f)