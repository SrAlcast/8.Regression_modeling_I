# 📈 Modelos de Regresión en Python

## 📖 Descripción

Este repositorio forma parte de una serie de ejercicios del Bootcamp de Hackio, con una finalidad educativa. Aquí exploramos diferentes modelos de regresión con datos codificados y técnicas comunes de preprocesamiento. Los objetivos principales son implementar, evaluar y comparar modelos para comprender su comportamiento y utilidad en problemas prácticos.

## 🗂️ Estructura del Proyecto

```
├── data/                # Conjuntos de datos en formato CSV
├── notebooks/           # Notebooks de Jupyter con ejemplos y análisis
├── src/                 # Scripts con funciones principales utilizadas en los notebooks
├── .gitattributes       # Configuraciones de Git
├── README.md            # Descripción del proyecto
```

- **`data/`**: Contiene los datos preprocesados y originales en formato CSV, utilizando codificaciones como label encoding, one-hot encoding y most frequent.
- **`notebooks/`**: Incluye notebooks educativos donde se implementan y visualizan los modelos de regresión.
- **`src/`**: Scripts reutilizables con funciones para preprocesamiento, entrenamiento de modelos y evaluación.

## 🚀 Modelos Implementados

Este repositorio incluye los siguientes modelos de regresión:
- **Regresión Lineal** ([LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html))
- **Árboles de Decisión para Regresión** ([DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html))

## 🛠️ Instalación y Requisitos

Este proyecto utiliza **Python 3.8+** y requiere las siguientes bibliotecas principales:

- [numpy](https://numpy.org/doc/)
- [pandas](https://pandas.pydata.org/docs/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [matplotlib](https://matplotlib.org/stable/contents.html)
- [seaborn](https://seaborn.pydata.org/)

## 📊 Visualizaciones y Métricas

El repositorio incluye análisis gráficos y comparaciones detalladas de rendimiento entre los modelos, utilizando métricas como:

- Error Cuadrático Medio (MSE)
- R-cuadrado (R²)
- Visualización de predicciones vs valores reales

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Si deseas colaborar, sigue estos pasos:

1. Realiza un fork del repositorio.
2. Crea una rama para tu contribución: `git checkout -b nombre-de-tu-rama`.
3. Realiza tus cambios y haz un commit: `git commit -m "Descripción de los cambios"`.
4. Envía un pull request.
 
