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


def log_regression(dataset_filenames, predict_file, single_variable):
    dataframes = [open_file(filename) for filename in dataset_filenames]

    inputs_data = pd.concat(dataframes, axis=1, join='outer')  # Combine all providefiles into a single dataframe
    inputs_data = inputs_data.loc[:, ~inputs_data.columns.duplicated()]  # Remove duplicate column 'Swap Recipient'

    # Clean input and split data into training data and test data
    x = inputs_data.drop(columns=['Swap Recipient'])  # Features (all material columns as predictors)
    y = inputs_data['Swap Recipient']  # Target variable (currency swap, 1: Yes, 0: No)
    if single_variable != '':
        x = x.loc[:, [single_variable]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Standardize data to improve model runtime
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Create Model
    # For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss
    model = LogisticRegression(solver='liblinear', C=0.05, multi_class='auto', penalty='l2')
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

    # Take test data from file and predict likelihood of currency swap
    predict(predict_file, scaler, model)


# Calculate the likelihood for new data points with scaled features assuming data is in same format as x_train
def predict(filename, scaler, model):
    data_to_predict = open_file(filename)
    scaled_data = scaler.transform(data_to_predict)
    probability_of_swap = model.predict_proba(scaled_data)[:, 1]  # Probability of receiving a currency swap (class 1)

    for index, row in data_to_predict.iterrows():
        int_index = data_to_predict.index.get_loc(index)
        print(f"{row.to_string()}\nProbability of receiving a currency swap: {probability_of_swap[int_index]}\n")


# TODO: Implement plot for single variable. Currently single variable is not working
def plot(single_variable):
    # Step 5: Plot the decision boundary and the data points
    # Note: As mentioned earlier, directly plotting the decision boundary for multiple features is not straightforward.
    # We can plot the contour plot to visualize the decision boundary and probabilities.

    # Define a meshgrid to plot the contour
    h = 0.01  # Step size in the mesh
    x_min, x_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1
    y_min, y_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Get predicted probabilities for the meshgrid
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # Plot the contour plot
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)

    # Plot the training data points
    plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap=plt.cm.RdBu, edgecolors='k')

    plt.xlabel(single_variable)
    plt.ylabel('Predict Probability of receiving a swap')
    plt.title('Logistic Regression Decision Boundary')
    plt.colorbar()
    plt.show()


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
    # List of prepared runtypes and their corresponding test file
    combined_run = ([INPUTS_DATA, EXPORTS_DATA], 'data/test_run.csv', '')
    inputs_run = ([INPUTS_DATA], 'data/inputs/test_run.csv', '')
    single_variable_manganese_run = ([INPUTS_DATA], 'data/inputs/test_run_manganese.csv', 'Manganese Ore')
    export_run = ([EXPORTS_DATA], 'data/exports/test_run.csv', '')

    # Select one of the runtypes above and add as an argument to the log_regression function
    # after the * to run the model on it
    log_regression(*combined_run)
