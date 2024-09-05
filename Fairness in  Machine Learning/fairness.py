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
dataset_adult_train, dataset_adult_test = dataset_adult.split([0.6], shuffle=True)
privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

min_max_scaler = MaxAbsScaler()
dataset_adult_train.features = min_max_scaler.fit_transform(dataset_adult_train.features)
dataset_adult_test.features = min_max_scaler.transform(dataset_adult_test.features)

# plain classifier without debiasing by using AdversialDebiasing
sess = tf.Session()
plain_model = AdversarialDebiasing(privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups,scope_name='plain_classifier',debias=False,sess=sess)
plain_model.fit(dataset_adult_train)

dataset_plain_test = plain_model.predict(dataset_adult_test)
classified_metric_plain_test = ClassificationMetric(dataset_adult_test,dataset_plain_test,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

sess.close()
tf.reset_default_graph()
sess = tf.Session()

# debiased classifier by using AdversialDebiasing
debiased_model = AdversarialDebiasing(privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups,scope_name='debiased_classifier',debias=True,sess=sess)
debiased_model.fit(dataset_adult_train)

dataset_debiasing_test = debiased_model.predict(dataset_adult_test)
classified_metric_debiasing_test = ClassificationMetric(dataset_adult_test,dataset_debiasing_test,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

# Calculating classification accuracy and other metrics
plain_TPR = classified_metric_plain_test.true_positive_rate()
plain_TNR = classified_metric_plain_test.true_negative_rate()
plain_model_classification_accuracy = 0.5*(plain_TPR+plain_TNR)
plain_model_equal_opportunity_difference = classified_metric_plain_test.equal_opportunity_difference()

debias_TPR = classified_metric_debiasing_test.true_positive_rate()
debias_TNR = classified_metric_debiasing_test.true_negative_rate()
debias_model_classification_accuracy = 0.5*(debias_TPR+debias_TNR)
debias_model_equal_opportunity_difference = classified_metric_debiasing_test.equal_opportunity_difference()

# Print the metrics
print("Plain Model Classification Accuracy:", plain_model_classification_accuracy)
print("Plain Model Equal Opportunity Difference:", plain_model_equal_opportunity_difference)
print("Debiased Model Classification Accuracy:", debias_model_classification_accuracy)
print("Debiased Model Equal Opportunity Difference:", debias_model_equal_opportunity_difference)
