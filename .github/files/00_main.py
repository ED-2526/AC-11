from a_local_feature_extraction import creation_of_descriptors, hola
from b_vocabulary_extraction import vocabulary
from c_convert_nontabular_to_tabular import dataset_to_histograms
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

creation_of_descriptors(['sift', 'harris', 'mser', 'dense'], flag = False)
K = 100
train_data, test_data, vocabulary  = vocabulary()

X_train, y_train = dataset_to_histograms(train_data, vocabulary, K)
X_test, y_test = dataset_to_histograms(test_data, vocabulary, K)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

clf = SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))