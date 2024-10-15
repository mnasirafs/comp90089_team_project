import pickle
import numpy as np
from data import construct_features
from high_impact_factors import feature_importance
from build_model import select_model, evaluate

# Paths can be parameterized or configured
PATH_TRAIN_SEQS = "./../../data/sepsis/processed_data/sepsis.seqs.train"
PATH_TRAIN_LABELS = "./../../data/sepsis/processed_data/sepsis.labels.train"
PATH_VALID_SEQS = "./../../data/sepsis/processed_data/sepsis.seqs.validation"
PATH_VALID_LABELS = "./../../data/sepsis/processed_data/sepsis.labels.validation"
PATH_TEST_SEQS = "./../../data/sepsis/processed_data/sepsis.seqs.test"
PATH_TEST_LABELS = "./../../data/sepsis/processed_data/sepsis.labels.test"

# Early stopping condition
EARLY_STOPPING_ROUNDS = 10

# Load datasets
print('===> Loading entire datasets')
train_seqs = pickle.load(open(PATH_TRAIN_SEQS, 'rb'))
y_train = pickle.load(open(PATH_TRAIN_LABELS, 'rb'))
valid_seqs = pickle.load(open(PATH_VALID_SEQS, 'rb'))
y_valid = pickle.load(open(PATH_VALID_LABELS, 'rb'))
test_seqs = pickle.load(open(PATH_TEST_SEQS, 'rb'))
y_test = pickle.load(open(PATH_TEST_LABELS, 'rb'))

# Construct features
X_train = construct_features(train_seqs)
X_valid = construct_features(valid_seqs)
X_test = construct_features(test_seqs)

# Combine training and validation data for model selection
X_train_valid = X_train + X_valid
y_train_valid = y_train + y_valid

# Perform feature importance analysis
feature_importance(X_train_valid, y_train_valid)

# Hyperparameter tuning with model selection
select_model(X_train_valid, y_train_valid)

# Evaluate models on test set
result = evaluate(X_train_valid, X_test, y_train_valid, y_test)