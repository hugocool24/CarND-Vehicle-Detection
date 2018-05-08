from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
import pickle
import glob
from features import extract_features

paths_non_vehicles = glob.glob('./non-vehicles/**/*.png', recursive=True)
print('Loading non-vehicle features')
non_vehicle_features = extract_features(paths_non_vehicles, cspace='YCrCb', hog_channel='ALL',
                                        spatial_feat=True, hist_feat=True, hog_feat=True)

paths_vehicles = glob.glob('./vehicles/**/*.png', recursive=True)
print('Loading vehicle features')
vehicle_features = extract_features(paths_vehicles, cspace='YCrCb', hog_channel='ALL',
                                    spatial_feat=True, hist_feat=True, hog_feat=True)

# Create label vector
y = np.hstack((np.ones(len(paths_vehicles)), np.zeros(len(paths_non_vehicles))))
# Stack the non car and car features together
X = vehicle_features + non_vehicle_features
X = np.asarray(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

# Normalize training data
# Fit a per-column scaler only on the training data
X_scaler = StandardScaler().fit(X_train)

with open('scaler.pkl', 'wb') as output:
    pickle.dump(X_scaler, output, pickle.HIGHEST_PROTOCOL)
# Apply the scaler to both X_train and X_test
scaled_X_train = X_scaler.transform(X_train)
scaled_X_test = X_scaler.transform(X_test)
# Use a linear SVC (support vector classifier)
svc = LinearSVC()
# Tune and train
parameters = {'C': [1, 10]}
clf = GridSearchCV(svc, parameters)

clf.fit(scaled_X_train, y_train)
print('Test Accuracy of SVC = ', clf.score(scaled_X_test, y_test))
print('Theese paramters were choosen: ', clf.best_params_)
# Save the classifier
with open('classifier.pkl', 'wb') as output:
    pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)
