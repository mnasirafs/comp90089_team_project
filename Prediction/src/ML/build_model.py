import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import GridSearchCV
import joblib

# Set a constant for random state
RANDOM_STATE = 41

# Function to perform model selection using GridSearchCV
def select_model(X, y):
    # Model names for identification
    model_names = ["DecisionTreeModel", "RandomForestModel", "AdaBoostModel", "MLPModel"]

    # Corresponding estimators for each model
    estimators = [
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        RandomForestClassifier(random_state=RANDOM_STATE),
        AdaBoostClassifier(random_state=RANDOM_STATE),
        MLPClassifier(max_iter=1000, random_state=RANDOM_STATE)
    ]

    # Hyperparameter grid for each model
    param_grids = [
        {'max_depth': range(1, 11)},  # DecisionTreeClassifier params
        {'n_estimators': range(10, 210, 10), 'max_depth': range(1, 11)},  # RandomForestClassifier params
        {'n_estimators': range(10, 310, 10), 'learning_rate': [0.001, 0.01, 0.1, 1]},  # AdaBoost params
        {'hidden_layer_sizes': [(10,), (50,), (100,), (10, 10), (10, 10, 10)]}  # MLPClassifier params
    ]

    # Create the directory if it doesn't exist
    output_dir = './../../out/selected_model_ml/'
    os.makedirs(output_dir, exist_ok=True)

    # Loop through models and apply GridSearchCV
    for i, estimator in enumerate(estimators):
        grid_search = GridSearchCV(estimator, param_grid=param_grids[i], scoring='roc_auc', cv=5)
        grid_search.fit(X, y)
        
        # Save the best model
        model_path = os.path.join(output_dir, f'{model_names[i]}.pkl')
        joblib.dump(grid_search.best_estimator_, model_path)
        
        # Print best score and parameters
        print(f"\nModel: {model_names[i]}")
        print(f"Best ROC-AUC score: {grid_search.best_score_:.3f}")
        print("Best parameters:")
        best_params = grid_search.best_estimator_.get_params()
        for param_name in sorted(param_grids[i].keys()):
            print(f"\t{param_name}: {best_params[param_name]}")

# Function to evaluate models on the test set
def evaluate(X_train, X_test, y_train, y_test):
    model_names = ["DecisionTreeModel", "RandomForestModel", "AdaBoostModel", "MLPModel"]
    evaluation_results = []

    # Load pre-trained models
    classifiers = [joblib.load(os.path.join('./../../out/selected_model_ml/', f'{name}.pkl')) for name in model_names]

    # Evaluate each model
    for name, clf in zip(model_names, classifiers):
        # Fit model on training data and predict on test data
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]  # Use probabilities for ROC-AUC

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        mcc = matthews_corrcoef(y_test, y_pred)

        # Store results
        evaluation_results.append([precision, recall, f1, mcc, roc_auc])

        # Print evaluation metrics for the current model
        print(f"\n{name} Evaluation:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        print(f"ROC-AUC: {roc_auc:.3f}")
        print(f"MCC: {mcc:.3f}")

    return evaluation_results
