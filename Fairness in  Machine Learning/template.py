

# Load all necessary packages
import sys
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from sklearn.preprocessing import MaxAbsScaler
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


# Get the dataset and split into train and test
dataset_adult = load_preproc_data_adult()
#use split function that's inbuilt for preloaded datasets like dataset.split([0.6], shuffle=True)
dataset_adult_train, dataset_adult_test = 
privileged_groups = [{'sex':1}]
unprivileged_groups = [{'sex':0}]

min_max_scaler = MaxAbsScaler()
dataset_adult_train.features = min_max_scaler.fit_transform(dataset_adult_train.features)
dataset_adult_test.features = min_max_scaler.transform(dataset_adult_test.features)

# plan classifier without debiasing by using AdversialDebiasing from inprocessing algorithms module and setting debias=False
# the parameters that are passed to AdversialDebiasing are privileged_groups, unprivileged_groups,scope_name,debias,sess
# sess is the session name of tf session that's been initially declared

sess = tf.Session()
plain_model = 
plain_model.fit(dataset_adult_train)

dataset_plain_test = plain_model.predict(dataset_adult_test)
# use ClassificationMetric from metrics module to get all the classification metric for the predicted and original test data
# the parameters that are needed for ClassificationMetric are test data,predicted data, unprivileged_groups, privileged_groups 
classified_metric_plain_test =
sess.close()
tf.reset_default_graph()
sess = tf.Session()

# debiased classifier by using AdversialDebiasing from inprocessing algorithms module and setting debias=True
debiased_model = 
debiased_model.fit(dataset_adult_train)

dataset_debiasing_test = debiased_model.predict(dataset_adult_test)


classified_metric_debiasing_test =

# print classification accuracy,  balanced classification accuracy, equal opportunity difference for both models to compare the difference
# you can use classification_accuracy() function to print classification accuracy
# balanced accuracy is the average of true postive rate and true negatve rate, use true_positive_rate() function to get true positive rate and true_negative_rate() to get true negative rate
# to calculate equal opportunity difference use equal_opportunity_difference() function
# the way to use these functions are classification_metric.function()
plain_model_equal_opportunity_difference=
plain_TPR = 
plain_TNR = 
plain_model_classification_accuracy=0.5*(plain_TPR+plain_TNR)
debias_TPR = 
debias_TNR = 
debias_model_classification_accuracy=0.5*(debias_TPR+debias_TNR)
debias_model_equal_opportunity_difference=

