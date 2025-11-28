from a_local_feature_extraction import creation_of_descriptors
from b_vocabulary_extraction import vocabulary
from c_convert_nontabular_to_tabular import dataset_to_histograms
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


creation_of_descriptors(['sift'], flag = False,
                        dim_descriptors=10, max_keypoints = 500)
K = 10
train_data, test_data, vocabulary  = vocabulary(flag_split = False, flag_kmeans = False, k= K)

X_train, y_train = dataset_to_histograms(train_data, vocabulary, K)
X_test, y_test = dataset_to_histograms(test_data, vocabulary, K)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

clf = svm.LinearSVC()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))