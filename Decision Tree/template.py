# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import matplotlib.pyplot as plt

# load dataset

pima_df = #YOUR CODE HERE

# Whenever dealing with any kind of data, try to see what the head values look like and a description of the data, using - head
pima_df.head()

# Also practice describing the data as you get valuble insights from it, using - describe methods

pima_df.describe()

# Additionally you may look at the shape and information about the data for better understanding of what you are dealing with, using - info and shape
pima_df.info()

shape = # YOUR CODE HERE
print(shape)

#Always no a null check as it gives us an idea of whether or not we need to impute or drop any sample(s)

pima_df.isnull().sum()

# Split dataset into features and target variable --> make sure you use the correct column names while subsetting for X and y variables
# Use Outcome as the target variable (i.e. y variable) and the rest as features (i.e. X variable)

# Features
X = # YOUR CODE HERE

# Target variable
y = # YOUR CODE HERE

# Split dataset into training set and test set - 80% traioning and 20% test --> try out other splits as well and check if you can get better results than the prescribed split of 80:20.
# Use random state as 1 but you are encouraged to play aropund with the values of random state as well

X_train, X_test, y_train, y_test = # YOUR CODE HERE

# Play around with the values of the DecisionTreeClassifier - change criterion and max_depth to find the best accuracy and report the same
# Prescribed parameters - criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5
clf = # YOUR CODE HERE

# Train and Fit Decision Tree Classifer
clf = # YOUR CODE HERE

#Predict the response for test dataset
dtree_y_pred = # YOUR CODE HERE

# Report the Accuracy, Recall, Precision and F1 score for the predictions that your classifier has made
# Use the metrics lib from sklearn previously imported. Also use 'weighted' as the average value for precision, recall and f1

dtree_y_true = y_test

accuracy = # Put the value of accuracy here -- 1.1 of Gradescope tests
precision = # Put the value of precision here -- 1.2 of Gradescope tests
recall = # Put the value of recall here -- 1.3 of Gradescope tests
f1_score = # Put the value of F1 score here -- 1.4 of Gradescope tests

print("Accuracy:", accuracy) # -- 1.1
print("Precision Score:", precision) # -- 1.2
print("Recall Score: ", recall) # -- 1.3
print("F1 Score: ", f1_score) # -- 1.4

# Here a template for writing the function for getting the tpr, fpr and threshold as well as AUC values is given as well as a simple way to plot the ROC curve
# you need to find out the predicted probabilities for the classifier you have written - using: redict_proba() function
# Pass these probabilities along with the y_true values into this function as specified and find the values mentioned above
# and report the AUC value and plot the ROC curve

from sklearn.metrics import roc_curve, auc

dtree_auc = 0

def plot_roc(dt_y_true, dt_probs):
    
    # Use sklearn.metrics.roc_curve() to get the values based on what the funciton returns - Read documentation here: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    
    dtree_fpr, dtree_tpr, threshold = # YOUR CODE HERE
    
    # Use sklearn.metrics.auc() to get the AUC score of your model - Read documentation here: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
    dtree_auc_val =  # Put the AUC score here -- 1.5 of Gradescope tests
    
    # Report the AUC score for the model you have created
    print('AUC=%0.2f'%dtree_auc_val) # -- 1.6
    
    
    # Plot the ROC curve using the probabilities and the true y values as passed into the fuction we have defined here
    
    plt.plot(dtree_fpr, dtree_tpr, label = 'AUC=%0.2f'%dtree_auc_val, color = 'darkorange')
    plt.legend(loc = 'lower right')
    plt.plot([0,1], [0,1], 'b--')
    plt.xlim([0,1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

    return dtree_auc_val

# We pass in the following parameters to the custom function we have written. For details as to why we do this - Read here: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score    

dtree_probs = clf.predict_proba(X_test) [:,1]
dtree_auc = plot_roc(dtree_y_true, dtree_probs) # 1.5 (this is where the value is returned by the function and verified)

# Report which cross validation value of k gave the best result as well as the max accuracy score - use StrafiedKFold Cross Validation method for 2 - 10 folds - Read about it here: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html#sklearn.model_selection.RepeatedStratifiedKFold

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

max_acc, max_k = 0, 0

for k in range(2, 11):

    skfold = #YOUR CODE HERE FOR: StratifiedKFold() - use random_state = 100, shuffle = True

    results_skfold_acc = (cross_val_score(clf, X, y, cv = skfold)).mean() * 100.0
    
    if results_skfold_acc > max_acc: # conditional check for getting max value and corresponding k value
        
    	# YOUR CODE HERE

    print("Accuracy: %.2f%%" % (results_skfold_acc))

best_accuracy =  # Put the accuracy score here from the values that you got -- 1.6 of Gradescope tests
best_k_fold =  # Put the value of k that gives the best accuracy here # -- 1.7 of Gradescope tests

print(best_accuracy, best_k_fold)