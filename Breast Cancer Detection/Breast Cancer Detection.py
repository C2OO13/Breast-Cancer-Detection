import sys
import numpy
import matplotlib
import pandas
import sklearn

print("Python: {}".format(sys.version))
print("Numpy: {}".format(numpy.__version__))
print("Matplotlib: {}".format(matplotlib.__version__))
print("Pandas: {}".format(pandas.__version__))
print("Sklearn: {}".format(sklearn.__version__))

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
cols = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion',
        'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
df = pd.read_csv(url, names=cols)
df.replace('?', -99999, inplace=True)  # This line will tell python to ignore all empty values
df.drop(["id"], 1, inplace=True)  # id is of no use
# axis = 1 for columns and 0 for rows
# print(df)
'''
print(df.info(), end="\n\n")
print(df.axes, end="\n\n")
print(df.shape, end="\n\n")

print(df.loc[399])
print(df.describe())
df.hist(figsize=(10, 10))
plt.show()

scatter_matrix(df, figsize=(18, 18))
plt.show()
'''
X = df.drop(['class'], 1, inplace=False)
Y = np.array(df['class'])

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

seed = 2  # Ensures the same intialization of random numbers take place so results are same everytime we run the algo
scoring = 'accuracy'

models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors=4)))
models.append(('SVM', SVC()))
models.append(('SVM Linear Kernel', LinearSVC()))
models.append(('SVM NuSVC', NuSVC()))

results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print("{0} : \n\tMean= {1} \n\tStandard Deviation= {2}".format(name, cv_results.mean(), cv_results.std()))


for name, model in models:
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    print(name)
    print(accuracy_score(Y_test, predictions))
    print(classification_report(Y_test, predictions, labels=np.unique(predictions)))
# Understand the difference between different SVM models
# LinearSVC is better scalable then SVC with Linear kernel and provides different results
# Because they use different methods to classify
# LinearSVC uses One vs All Classification
# SVC on the other hand uses One vs One classification

clf = KNeighborsClassifier()
# clf = SVC()
# clf = LinearSVC()
# clf = NuSVC()
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print(accuracy)
# Manually trying the models
example_measures = np.array([[4, 2, 1, 10, 5, 2, 3, 2, 7]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)