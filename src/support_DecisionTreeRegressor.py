
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
from sklearn.model_selection import KFold,LeaveOneOut, cross_val_score
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

def graficar_arbol_decision(modelo, nombres_caracteristicas, tamano_figura=(30, 30), tamano_fuente=12):
    """
    Grafica un árbol de decisión con opciones personalizables.

    Parámetros:
        modelo: Árbol de decisión entrenado (DecisionTreeClassifier o DecisionTreeRegressor).
        nombres_caracteristicas: Lista o índice con los nombres de las características (columnas).
        tamano_figura: Tuple, tamaño de la figura (ancho, alto).
        tamano_fuente: Tamaño de la fuente en la gráfica.
    """
    plt.figure(figsize=tamano_figura)
    plot_tree(
        decision_tree=modelo,
        feature_names=nombres_caracteristicas,
        filled=True,  # Colorear los nodos
        rounded=True,  # Esquinas redondeadas
        fontsize=tamano_fuente,
        proportion=True,  # Mostrar proporciones en lugar de valores absolutos
        impurity=False  # Ocultar impureza de los nodos
    )
    plt.show()

def loo_cross_validation_rmse(model, X, y):
    """
    Realiza validación cruzada Leave-One-Out (LOO) y calcula el RMSE promedio.

    Parámetros:
        model: modelo de regresión (ej. LinearRegression de sklearn)
        X: DataFrame o matriz con las características (features)
        y: Serie o vector con el objetivo (target)
    
    Retorno:
        float: RMSE promedio obtenido en la validación cruzada
    """
    loo = LeaveOneOut()
    scores = []

    for train_index, test_index in tqdm(loo.split(X), total=len(X)):
        # Dividir los datos en entrenamiento y prueba
        X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
        y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]

        # Entrenar el modelo y predecir
        model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_test_cv)

        # Calcular RMSE
        rmse = np.sqrt(mean_squared_error([y_test_cv.values[0]], y_pred))
        scores.append(rmse)

    # Calcular y retornar el RMSE promedio
    return np.mean(scores)
