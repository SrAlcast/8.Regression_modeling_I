
# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np
np.set_printoptions(formatter={'float_kind': lambda x: f"{float(x):.4f}"})

# Visualizaciones
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

# Para realizar la regresión lineal y la evaluación del modelo
# -----------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold,LeaveOneOut, cross_val_scor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def metricas(y_train,y_train_pred,y_test,y_test_pred):
    metricas = {
        'train': {
            'r2_score': r2_score(y_train, y_train_pred),
            'MAE': mean_absolute_error(y_train, y_train_pred),
            'MSE': mean_squared_error(y_train, y_train_pred),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred))
        },
        'test': {
            'r2_score': r2_score(y_test, y_test_pred),
            'MAE': mean_absolute_error(y_test, y_test_pred),
            'MSE': mean_squared_error(y_test, y_test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred))

        },
        'diferences': {
            'r2_score': (r2_score(y_train, y_train_pred))-(r2_score(y_test, y_test_pred)),
            'MAE': (mean_absolute_error(y_train, y_train_pred))-(mean_absolute_error(y_test, y_test_pred)),
            'MSE': (mean_squared_error(y_train, y_train_pred))- (mean_squared_error(y_test, y_test_pred)),
            'RMSE': (np.sqrt(mean_squared_error(y_train, y_train_pred)))-(np.sqrt(mean_squared_error(y_test, y_test_pred)))

        }
    }
    return pd.DataFrame(metricas).T

import matplotlib.pyplot as plt
import seaborn as sns

def plot_real_vs_predicted(y_test, y_test_pred, y_train, y_train_pred):
    """
    Plots Real vs Predicted Prices for test and train datasets.

    Parameters:
        y_test (array-like): Actual target values for the test set.
        y_test_pred (array-like): Predicted target values for the test set.
        y_train (array-like): Actual target values for the train set.
        y_train_pred (array-like): Predicted target values for the train set.
    """
    plt.figure(figsize=(10, 6), dpi=150)
    plt.suptitle('Real vs. Predicted Prices')

    # Test data plot
    plt.subplot(2, 1, 1)
    sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.6, s=10, label="Test data")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Perfect prediction line", lw=0.7)
    plt.xlabel('Real Prices (y_test)')
    plt.ylabel('Predicted Prices (y_test_pred)')
    plt.legend()

    # Train data plot
    plt.subplot(2, 1, 2)
    sns.scatterplot(x=y_train, y=y_train_pred, alpha=0.6, s=10, color="forestgreen", label="Train data")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Perfect prediction line", lw=0.7)
    plt.xlabel('Real Prices (y_train)')
    plt.ylabel('Predicted Prices (y_train_pred)')
    plt.legend()

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def plot_residuals(y_test, y_test_pred, y_train, y_train_pred):
    """
    Plots residual plots (absolute and relative) for test and train datasets.

    Parameters:
        y_test (array-like): Actual target values for the test set.
        y_test_pred (array-like): Predicted target values for the test set.
        y_train (array-like): Actual target values for the train set.
        y_train_pred (array-like): Predicted target values for the train set.
    """
    plt.figure(figsize=(10, 6), dpi=150)
    plt.suptitle('Residual Plot (Absolute and Relative)')

    # Absolute residuals for test data
    residuals_test = y_test_pred - y_test
    plt.subplot(2, 2, 1)
    sns.scatterplot(x=y_test, y=residuals_test, alpha=0.6, s=10, label="Test Data")
    plt.axhline(0, color='red', linestyle='--', label="Perfect prediction", lw=0.7)
    plt.xlabel('Real Prices (y_test)')
    plt.ylabel('Residuals')
    plt.legend()

    # Absolute residuals for train data
    residuals_train = y_train_pred - y_train
    plt.subplot(2, 2, 2)
    sns.scatterplot(x=y_train, y=residuals_train, alpha=0.6, s=10, color="forestgreen", label="Train Data")
    plt.axhline(0, color='red', linestyle='--', label="Perfect prediction", lw=0.7)
    plt.xlabel('Real Prices (y_train)')
    plt.ylabel('Residuals')
    plt.legend()

    # Relative residuals (%) for test data
    relative_residuals_test = (y_test_pred - y_test) / y_test * 100
    plt.subplot(2, 2, 3)
    sns.scatterplot(x=y_test, y=relative_residuals_test, alpha=0.6, s=10, label="Test Data")
    plt.axhline(0, color='red', linestyle='--', label="Perfect prediction", lw=0.7)
    plt.xlabel('Real Prices (y_test)')
    plt.ylabel('Residuals (%)')
    plt.legend()

    # Relative residuals (%) for train data
    relative_residuals_train = (y_train_pred - y_train) / y_train * 100
    plt.subplot(2, 2, 4)
    sns.scatterplot(x=y_train, y=relative_residuals_train, alpha=0.6, s=10, color="forestgreen", label="Train Data")
    plt.axhline(0, color='red', linestyle='--', label="Perfect prediction", lw=0.7)
    plt.xlabel('Real Prices (y_train)')
    plt.ylabel('Residuals (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()
