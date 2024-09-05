import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

pima_df = pd.read_csv('diabetes.csv')
pima_df.head()
pima_df.describe()
pima_df.info()

shape = pima_df.shape
print(shape)

pima_df.isnull().sum()

X = pima_df.drop(columns=['Outcome'])
y = pima_df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
clf = clf.fit(X_train, y_train)

dtree_y_pred = clf.predict(X_test)
dtree_y_true = y_test

accuracy = metrics.accuracy_score(y_test, dtree_y_pred) # Put the value of accuracy here -- 1.1 of Gradescope tests
precision = metrics.precision_score(y_test, dtree_y_pred, average='weighted') # Put the value of precision here -- 1.2 of Gradescope tests
recall = metrics.recall_score(y_test, dtree_y_pred, average='weighted') # Put the value of recall here -- 1.3 of Gradescope tests
f1_score = metrics.f1_score(y_test, dtree_y_pred, average='weighted') # Put the value of F1 score here -- 1.4 of Gradescope tests

print("Accuracy:", accuracy) # -- 1.1
print("Precision Score:", precision) # -- 1.2
print("Recall Score: ", recall) # -- 1.3
print("F1 Score: ", f1_score) # -- 1.4

from sklearn.metrics import roc_curve, auc
dtree_auc = 0


def plot_roc(dt_y_true, dt_probs):
    dtree_fpr, dtree_tpr, threshold = roc_curve(dt_y_true, dt_probs)
    dtree_auc_val = auc(dtree_fpr, dtree_tpr) # Put the AUC score here -- 1.5 of Gradescope tests
    print('AUC=%0.2f'%dtree_auc_val) # -- 1.6

    # Plot the ROC curve
    plt.plot(dtree_fpr, dtree_tpr, label='AUC=%0.2f' % dtree_auc_val, color='darkorange')
    plt.legend(loc='lower right')
    plt.plot([0,1], [0,1], 'b--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

    return dtree_auc_val

dtree_probs = clf.predict_proba(X_test) [:,1]
dtree_auc = plot_roc(dtree_y_true, dtree_probs) # 1.5 (this is where the value is returned by the function and verified)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

max_acc, max_k = 0, 0

for k in range(2, 11):
    skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=100) # YOUR CODE HERE FOR: StratifiedKFold() - use random_state = 100, shuffle = True
    results_skfold_acc = (cross_val_score(clf, X, y, cv=skfold)).mean() * 100.0
    if results_skfold_acc > max_acc:  # conditional check for getting max value and corresponding k value
        max_acc = results_skfold_acc
        max_k = k

    print("Accuracy: %.2f%%" % results_skfold_acc)

best_accuracy = max_acc  # Assign the maximum accuracy to best_accuracy
best_k_fold = max_k  # Assign the value of k that gives the best accuracy to best_k_fold

print(best_accuracy, best_k_fold)