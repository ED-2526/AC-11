from sklearn.decomposition import PCA
import cv2
import numpy as np
import os
import pickle

def process_image_channels(img, submethod):
    """
    Procesa la imagen según el submethod y devuelve lista de canales a procesar.
    - grey: devuelve [imagen_gris]
    - color: devuelve [imagen_gris] pero los keypoints se detectan en gris
    - splitted: devuelve [canal_B, canal_G, canal_R]
    """
    if submethod == 'grey':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return [gray]
    elif submethod == 'color':
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, _, _ = cv2.split(hsv)
        return [h]
    elif submethod == 'splitted':
        # Separar en canales B, G, R
        b, g, r = cv2.split(img)
        return [r, g, b]  # Devolvemos en orden R, G, B
    else:
        raise ValueError(f"Submethod '{submethod}' no reconocido. Usa 'grey', 'color' o 'splitted'")


def extract_sift_features(image_path, max_keypoints, submethod='grey'):
    img = cv2.imread(str(image_path))
    channels = process_image_channels(img, submethod)
    
    sift = cv2.SIFT_create(nfeatures=max_keypoints if max_keypoints else 0)
    
    if submethod == 'splitted':
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints = sift.detect(gray, None)
        
        if not keypoints:
            return None
        
        
        all_descriptors = []
        for channel in channels:
            _, descriptors = sift.compute(channel, keypoints)
            if descriptors is not None:
                all_descriptors.append(descriptors)
        
        if len(all_descriptors) != 3:
            return None
        
        return np.hstack(all_descriptors)
    
    else:
        keypoints, descriptors = sift.detectAndCompute(channels[0], None)
        return descriptors


def extract_harris_sift(image_path, max_keypoints, submethod='grey'):
    img = cv2.imread(str(image_path))
    channels = process_image_channels(img, submethod)
    
    sift = cv2.SIFT_create()
    
    if submethod == 'splitted':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        max_corners = max_keypoints if max_keypoints else 0
        corners = cv2.goodFeaturesToTrack(gray, max_corners, 0.01, 10)
        
        if corners is None:
            return None
        
        keypoints = [cv2.KeyPoint(x=c[0][0], y=c[0][1], size=20) for c in corners]
        
        all_descriptors = []
        for channel in channels:
            _, descriptors = sift.compute(channel, keypoints)
            if descriptors is not None:
                all_descriptors.append(descriptors)
        
        if len(all_descriptors) != 3:
            return None
        
        return np.hstack(all_descriptors)
    
    else:
        channel = channels[0]
        max_corners = max_keypoints if max_keypoints else 0
        corners = cv2.goodFeaturesToTrack(channel, max_corners, 0.01, 10)
        
        if corners is None:
            return None
        
        keypoints = [cv2.KeyPoint(x=c[0][0], y=c[0][1], size=20) for c in corners]
        keypoints, descriptors = sift.compute(channel, keypoints)
        return descriptors


def extract_dense_sift(image_path, max_keypoints, submethod='grey', step_size=10, patch_size=16):
    img = cv2.imread(str(image_path))
    
    if img is None:
        print(f"  [WARNING] No se pudo cargar la imagen: {image_path}")
        return None
    
    if img.shape[0] < patch_size or img.shape[1] < patch_size:
        print(f"  [WARNING] Imagen demasiado pequeña ({img.shape}): {image_path}")
        return None
    
    channels = process_image_channels(img, submethod)
    sift = cv2.SIFT_create()
    
    if submethod == 'splitted':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        if h <= patch_size or w <= patch_size:
            return None
        
        keypoints = []
        for y in range(0, h - patch_size, step_size):
            for x in range(0, w - patch_size, step_size):
                keypoints.append(cv2.KeyPoint(x, y, patch_size))
        
        if len(keypoints) == 0:
            return None
        
        if max_keypoints and len(keypoints) > max_keypoints:
            indices = np.linspace(0, len(keypoints) - 1, max_keypoints, dtype=int)
            keypoints = [keypoints[i] for i in indices]
        
        all_descriptors = []
        for channel in channels:
            try:
                _, descriptors = sift.compute(channel, keypoints)
                if descriptors is not None:
                    all_descriptors.append(descriptors)
            except Exception as e:
                print(f"  [WARNING] Error procesando {image_path}: {e}")
                continue
        
        if len(all_descriptors) != 3:
            return None
        
        return np.hstack(all_descriptors)
    
    else:
        channel = channels[0]
        h, w = channel.shape
        
        if h <= patch_size or w <= patch_size:
            return None
        
        keypoints = []
        for y in range(0, h - patch_size, step_size):
            for x in range(0, w - patch_size, step_size):
                keypoints.append(cv2.KeyPoint(x, y, patch_size))
        
        if len(keypoints) == 0:
            return None
        
        if max_keypoints and len(keypoints) > max_keypoints:
            indices = np.linspace(0, len(keypoints) - 1, max_keypoints, dtype=int)
            keypoints = [keypoints[i] for i in indices]
        
        try:
            keypoints, descriptors = sift.compute(channel, keypoints)
            return descriptors
        except Exception as e:
            print(f"  [WARNING] Error procesando {image_path}: {e}")
            return None

def extract_features_from_dataset(data_dir, max_keypoints, dim_descriptors, num_classes, method):
    uppermethod, submethod = method.split(',')
    methods = {
        'sift': extract_sift_features,
        'harris': extract_harris_sift,
        'dense': extract_dense_sift
    }

    extract_fn = methods[uppermethod]
    result = {}

    food_names = sorted(os.listdir(data_dir))
    if num_classes is not None and num_classes > 0:
        food_names = food_names[:num_classes]
        print(f"Limitando a {num_classes} clases: {food_names}")
    
    for food_name in food_names:
        print(f"Tratando con {food_name}")
        class_path = os.path.join(data_dir, food_name)
        
        if not os.path.isdir(class_path):
            continue
        
        result[food_name] = {}    
        for img_name in os.listdir(class_path):
            if not img_name.endswith(('.jpg', '.png', '.jpeg')):
                continue
                
            img_path = os.path.join(class_path, img_name)
            descriptors = extract_fn(img_path, max_keypoints, submethod)
            
            if descriptors is not None and len(descriptors) > 0:
                result[food_name][img_path] = descriptors

    if dim_descriptors == 0:
        print("No apliquem PCA, dim_descriptors -> 0")
    else:
        all_descriptors = []
        for food_name in result:
            for img_path in result[food_name]:
                descriptors = result[food_name][img_path]
                all_descriptors.append(descriptors)
        
        all_descriptors = np.vstack(all_descriptors)
        print(f"\nTotal de descriptors recolectados: {all_descriptors.shape[0]}")
        print(f"Dimensionalidad original: {all_descriptors.shape[1]}D")
        
        pca = PCA(n_components=dim_descriptors)
        pca.fit(all_descriptors)

        for food_name in result:
            print(f"\nTransformant: {food_name}")
            for img_path in result[food_name]:
                descriptors_original = result[food_name][img_path]
                descriptors_reduced = pca.transform(descriptors_original)
                result[food_name][img_path] = descriptors_reduced
    
    return result

def creation_of_descriptors(methods = ['sift'], flag = False, dim_descriptors = 64, max_keypoints = 500, num_classes = None):
    if flag == False:
        print("No s'ha executat la extracció perquè no es vol" \
        "sobreescriure lo queja hi ha un pickle")
        return
    else:
        dir = os.path.join(os.path.dirname(__file__), '..', 'Food Classification')
        for method in methods:
            result = extract_features_from_dataset(dir, max_keypoints,  dim_descriptors, num_classes, method)
            with open(os.path.join(os.path.dirname(__file__), f'features{method}_dim{dim_descriptors}_maxkeypoints{max_keypoints}.pickle'), 'wb') as f:
                pickle.dump(result, f)