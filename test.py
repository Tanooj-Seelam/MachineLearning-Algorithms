import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_roc_curve

# Load your dataset (replace 'heart1.csv' with your actual dataset)
data = pd.read_csv('heart1.csv')

# Split the data into features (X) and target variable (y)
X = data.drop(columns='a1p2', axis=1)
y = data['a1p2']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# Standardize the features
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Perceptron Classifier
percep = Perceptron(max_iter=10, tol=1e-3, eta0=0.001, fit_intercept=True, random_state=0)
percep.fit(X_train_std, y_train)

# Logistic Regression
c = 0.3
lr = LogisticRegression(C=c, random_state=0)
lr.fit(X_train_std, y_train)

# Support Vector Machine
c = 0.25
svm = SVC(kernel='linear', C=c, random_state=2)
svm.fit(X_train_std, y_train)

# Decision Tree Classifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=2)
tree.fit(X_train, y_train)

# Random Forest Classifier
forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)

# K-Nearest Neighbors
neighs = 8
knn = KNeighborsClassifier(n_neighbors=neighs, p=2, metric='minkowski', weights='distance')
knn.fit(X_train_std, y_train)

# Evaluate and compare model performances
models = [percep, lr, svm, tree, forest, knn]

for model in models:
    model_name = model._class.name_
    y_train_pred = model.predict(X_train_std)
    y_test_pred = model.predict(X_test_std)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"{model_name}:")
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print()

# Plot ROC curves
fig, ax = plt.subplots(figsize=(8, 6))
for model in models:
    plot_roc_curve(model, X_test_std, y_test, ax=ax, alpha=0.7)  # Use X_test_std here
plt.legend(models)
plt.show()