import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt

df_rf = pd.read_csv('diabetes.csv')
df_rf.head()
df_rf.isnull().sum()

X = df_rf.drop('Outcome', axis=1)
y = df_rf['Outcome']
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

rf_clf = RandomForestClassifier(criterion='gini', min_samples_leaf=1, min_samples_split=10, max_features='auto', random_state=1)

rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)

rf_y_true = y_test

accuracy = accuracy_score(rf_y_true, y_pred)
precision = precision_score(rf_y_true, y_pred, average='weighted')
recall = recall_score(rf_y_true, y_pred, average='weighted')
f1_score = f1_score(rf_y_true, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision Score:", precision)
print("Recall Score: ", recall)
print("F1 Score: ", f1_score)

max_acc, max_k = 0, 0


def plot_roc(rf_y_true, rf_probs):

    rf_fpr, rf_tpr, rf_threshold = roc_curve(rf_y_true, rf_probs)

    rf_auc_val = auc(rf_fpr, rf_tpr)

    print('AUC=%0.2f' % rf_auc_val)

    plt.plot(rf_fpr, rf_tpr, label='AUC=%0.2f' % rf_auc_val, color='darkorange')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'b--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

    return rf_auc_val


rf_probs = rf_clf.predict_proba(X_test)[:, 1]
rf_auc = plot_roc(rf_y_true, rf_probs)

# Inside the loop for k-fold validation
for k in range(2, 11):
    skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=100)
    results_skfold_acc = (cross_val_score(rf_clf, X, y, cv=skfold)).mean() * 100.0

    if results_skfold_acc > max_acc:
        max_acc = results_skfold_acc
        max_k = k

best_accuracy = max_acc
best_k_fold = max_k

print(best_accuracy, best_k_fold)
