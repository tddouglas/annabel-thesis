import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # For plotting data

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


def plot_grid(grid):
    result_df = pd.DataFrame.from_dict(grid.cv_results_, orient='columns')
    print(result_df.columns)
    sns.relplot(data=result_df,
                kind='line',
                x='param_classifier__n_neighbors',
                y='mean_test_score',
                hue='param_scaler',
                col='param_classifier__p')
    plt.show()

    sns.relplot(data=result_df,
                kind='line',
                x='param_classifier__n_neighbors',
                y='mean_test_score',
                hue='param_scaler',
                col='param_classifier__leaf_size')
    plt.show()