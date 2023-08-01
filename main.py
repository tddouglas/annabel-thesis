#!/usr/bin/python3
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, PowerTransformer, MaxAbsScaler, LabelEncoder

from model_tester import optimize_params, evaluate_model
from open_file import open_csv

np.set_printoptions(threshold=sys.maxsize)
INPUTS_DATA = "data/inputs/imports.csv"
EXPORTS_DATA = "data/exports/exports.csv"
CONTROLS_DATA = "data/controls/controls.csv"


def create_dataset(dataset_filenames, single_variable):
    dataframes = [open_csv(filename) for filename in dataset_filenames]
    if len(dataset_filenames) > 1:
        inputs_data = pd.concat(dataframes, axis=1, join='inner')
        inputs_data = inputs_data.loc[:, ~inputs_data.columns.duplicated()]  # Remove duplicate column 'Swap Recipient'
    else:
        inputs_data = dataframes[0]

    # Clean input and split data into independent and dependent variables (x and y)
    x = inputs_data.drop(columns=['Swap Recipient'])
    y = inputs_data['Swap Recipient']  # Target variable (currency swap, 1: Yes, 0: No)
    if single_variable != '':
        x = x.loc[:, [single_variable]]

    return x, y


# Single model run against best performing model
def run_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Run best performing pipeline from optimization run
    pipe = Pipeline(steps=[('scaler', StandardScaler()),
                           ('classifier', RandomForestClassifier(n_estimators=100))])
    pipe.fit(x_train, y_train)

    evaluate_model(pipe, x_test, y_test)
    return pipe


# Calculate the likelihood for new data points with scaled features assuming data is in same format as x_train
def predict(filename, scaler, model):
    data_to_predict = open_csv(filename)
    scaled_data = scaler.transform(data_to_predict)
    probability_of_swap = model.predict_proba(scaled_data)[:, 1]  # Probability of receiving a currency swap (class 1)

    for index, row in data_to_predict.iterrows():
        int_index = data_to_predict.index.get_loc(index)
        print(f"{row.to_string()}\nProbability of receiving a currency swap: {probability_of_swap[int_index]}\n")


if __name__ == '__main__':
    # List of prepared input file combos
    inputs = [INPUTS_DATA]
    export = [EXPORTS_DATA]
    controls_run = [CONTROLS_DATA]
    inputs_control = [INPUTS_DATA, CONTROLS_DATA]
    exports_control = [EXPORTS_DATA, CONTROLS_DATA]
    combined_run = [INPUTS_DATA, EXPORTS_DATA, CONTROLS_DATA]

    # Select one of the above file combos. Pass in a single column header as the second param if you want to
    # Run in "Single File Mode" only evaluating that column from the dataset
    x, y = create_dataset(inputs_control, '')

    if len(sys.argv) > 1:
        optimize = sys.argv[1]
        if optimize == 'optimize':
            optimize_params(x, y)
    else:
        # Run Best Model as output from optimize run above
        pipe = run_model(x, y)

        # Take test data from file and predict likelihood of currency swap
        # predict('data/test_run.csv', pipe)
