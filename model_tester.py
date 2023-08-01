from pandas import read_csv  # For dataframes
from numpy import ravel  # For matrices
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split  # For train/test splits
from sklearn.neighbors import KNeighborsClassifier  # The k-nearest neighbor classifier
from sklearn.feature_selection import VarianceThreshold  # Feature selector
from sklearn.pipeline import Pipeline  # For setting up pipeline
# Various pre-processing steps
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, PowerTransformer, MaxAbsScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV  # For optimization
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score
import pandas as pd


def optimize_params(x, y):

    # Define a list of scalers to test
    scalers = [
        ('StandardScaler', StandardScaler()),
        ('MinMaxScaler', MinMaxScaler()),
        ('Normalizer', Normalizer()),
        ('MaxAbsScaler', MaxAbsScaler())
    ]

    # Define a dictionary of classifiers and their corresponding hyperparameter grids
    classifiers = {
        'Naive Bayes': (GaussianNB(), {}),
        'Decision Tree': (DecisionTreeClassifier(), {'classifier__criterion': ['gini', 'entropy']}),
        'Random Forest': (RandomForestClassifier(), {'classifier__n_estimators': [50, 100, 200]}),
        'SVM': (SVC(probability=True), {'classifier__kernel': ['linear', 'rbf']}),
        'k-NN': (KNeighborsClassifier(), {'classifier__n_neighbors': [3, 5, 7]}),
        'Logistic Regression': (
            LogisticRegression(),
            {'classifier__solver': ['lbfgs', 'liblinear', 'newton-cg'], 'classifier__penalty': ['l2'],
             'classifier__C': [0.1, 1.0, 10.0]}),
    }

    # Define the range of values for K (number of features to select) and threshold (for SelectKBest)
    k_values = [5, 10, 15]
    thresholds = [0, 0.25, 0.5]

    # Maybe test for PCA for dimensionality reduction
    # pca = PCA(n_components=0.70)  # Choose the number of components that explain 95% of the variance
    # x_train = pca.fit_transform(x_train)
    # x_test = pca.transform(x_test)

    # Perform GridSearchCV to find the best scaler, classifier, and parameters
    best_pipeline = {'Naive Bayes': {}, 'Decision Tree': {}, 'Random Forest': {}, 'SVM': {}, 'k-NN': {}, 'Logistic Regression': {}}

    for i in range(5):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=i)
        for scaler_name, scaler in scalers:
            for classifier_name, (classifier, params) in classifiers.items():
                pipeline = Pipeline([
                    ('scaler', scaler),
                    ('classifier', classifier),
                ])
                grid_search = GridSearchCV(pipeline, params, cv=5, scoring='f1', n_jobs=-1)
                grid_search.fit(x_train, y_train)
                print(f"Evaluating{grid_search.best_estimator_}\n")
                y_pred = grid_search.predict(x_test)
                f1 = f1_score(y_test, y_pred)

                if str(grid_search.best_params_) in best_pipeline[classifier_name]:
                    best_pipeline[classifier_name][str(grid_search.best_params_)].append(f1)
                else:
                    best_pipeline[classifier_name][str(grid_search.best_params_)] = [f1]

    for key, value in best_pipeline.items():
        print(f"{key} model results:")
        for params, f1_scores in value.items():
            average = sum(f1_scores) / len(f1_scores)
            if average > 0.40:
                print(f"{params} with {len(f1_scores)} entries: {average}")


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)

    # Calculate accuracy, precision, recall, and F1-score (optional)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    probabilities = model.predict_proba(x_test)[:, 1]
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")

    # Print test swap results vs model prediction data
    y_pred_probabilities = pd.Series(probabilities).set_axis(y_test.index)
    y_pred_series = pd.Series(y_pred).set_axis(y_test.index)
    pred_test_compare = pd.concat([y_test, y_pred_series, y_pred_probabilities], axis=1)
    print(
        f"y_test vs. y_pred results:\n{pred_test_compare.set_axis(['y_test', 'y_pred', 'y_pred probabilities'], axis=1)}")
