#!/usr/bin/python3
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.set_printoptions(threshold=sys.maxsize)
INPUTS_DATA = "data/inputs/sheet1_green_inputs_revised.csv"
EXPORTS_DATA = "data/exports/sheet2_green_exports_revised.csv"

TEST_DATA = "data/inputs/test_run.csv"


def log_regression(dataset_filenames):
    dataframes = [open_file(filename) for filename in dataset_filenames]

    inputs_data = pd.concat(dataframes, axis=1, join='outer')  # Combine all provided files into a single dataframe
    inputs_data = inputs_data.loc[:, ~inputs_data.columns.duplicated()]  # Remove duplicate column 'Swap Recipient'

    # Clean input and split data into training data and test data
    x = inputs_data.drop(columns=['Swap Recipient'])  # Features (all material columns as predictors)
    y = inputs_data['Swap Recipient']  # Target variable (currency swap, 1: Yes, 0: No)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Standardize data to improve model runtime
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Create model and train it
    model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr',
                               random_state=0)
    model.fit(x_train, y_train)

    # Evaluate Model
    y_pred = model.predict(x_test)

    # Calculate accuracy, precision, recall, and F1-score (optional)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")

    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print(f"Training score {train_score}")
    print(f"Test score {test_score}\n")

    # Take test data from file and predict likelihood of currency swap
    predict(TEST_DATA, scaler, model)


# Calculate the likelihood for new data points with scaled features assuming data is in same format as x_train
def predict(filename, scaler, model):
    data_to_predict = open_file(filename)
    scaled_data = scaler.transform(data_to_predict)
    probability_of_swap = model.predict_proba(scaled_data)[:, 1]  # Probability of receiving a currency swap (class 1)

    for index, row in data_to_predict.iterrows():
        int_index = data_to_predict.index.get_loc(index)
        print(f"{row.to_string()}\nProbability of receiving a currency swap: {probability_of_swap[int_index]}\n")


# Converter used when importing Pandas data set to turn string percentage columns -> floats
def p2f(x):
    if x == '-':
        return float(0)
    else:
        return float(x.strip('%')) / 100


# Helper to open file and convert to Pandas DataFrame
def open_file(filename):
    converter_mapping = {"Manganese Ore": p2f, "Copper Ore": p2f, "Nickel Ore": p2f, "Cobalt Ore": p2f, "Zinc Ore": p2f,
                         "Chromium Ore": p2f, "Molybdenum Ore": p2f, "Rare Earth Metals": p2f, "Natural Graphite": p2f,
                         "Artificial Graphite": p2f, "Lithium Oxide": p2f, "Silicon": p2f,
                         "Semiconductor devices": p2f, "Electric motors": p2f, "Electric parts": p2f,
                         "Secondary batteries": p2f, "Steam turbines": p2f, "Hydraulic turbines": p2f,
                         "Gas turbines": p2f, "Electrolysers": p2f}
    try:
        with open(filename, 'r', encoding='utf-8-sig') as f:
            return pd.read_csv(f, dtype={"Country Name": "string"}, converters=converter_mapping,
                               index_col="Country Name")
    # except KeyError as e:
    #     print(f"Key Error {str(e)}")
    finally:
        f.close()


if __name__ == '__main__':
    files = [INPUTS_DATA]  # List of Datasets to include in logit
    log_regression(files)
