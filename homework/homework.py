# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import gzip
import pickle
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

def save_model_gzip(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f)


def write_metrics_jsonlines(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

grading_dir = "files/grading"
use_grading = os.path.exists(os.path.join(grading_dir, "x_train.pkl"))

if use_grading:
    with open(os.path.join(grading_dir, "x_train.pkl"), "rb") as f:
        X_train = pickle.load(f)
    with open(os.path.join(grading_dir, "y_train.pkl"), "rb") as f:
        y_train = pickle.load(f)
    with open(os.path.join(grading_dir, "x_test.pkl"), "rb") as f:
        X_test = pickle.load(f)
    with open(os.path.join(grading_dir, "y_test.pkl"), "rb") as f:
        y_test = pickle.load(f)
else:
    train = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
    test = pd.read_csv("files/input/test_data.csv.zip", compression="zip")

    train = train.rename(columns={"default payment next month": "default"})
    test = test.rename(columns={"default payment next month": "default"})
    if "ID" in train.columns:
        train = train.drop(columns=["ID"])
    if "ID" in test.columns:
        test = test.drop(columns=["ID"])

    train = train.replace(" ", np.nan).dropna()
    test = test.replace(" ", np.nan).dropna()

    if "EDUCATION" in train.columns:
        train["EDUCATION"] = train["EDUCATION"].apply(lambda x: x if x <= 4 else 4)
    if "EDUCATION" in test.columns:
        test["EDUCATION"] = test["EDUCATION"].apply(lambda x: x if x <= 4 else 4)

    X_train = train.drop(columns=["default"])
    y_train = train["default"]
    X_test = test.drop(columns=["default"])
    y_test = test["default"]

if not isinstance(X_train, pd.DataFrame):
    X_train = pd.DataFrame(X_train)
if not isinstance(X_test, pd.DataFrame):
    X_test = pd.DataFrame(X_test)

candidates_cat = ["SEX", "EDUCATION", "MARRIAGE"]
categorical = [c for c in candidates_cat if c in X_train.columns]
numeric = [c for c in X_train.columns if c not in categorical]

transformers = []
if categorical:
    transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
if numeric:
    transformers.append(("num", MinMaxScaler(), numeric))

preprocessor = ColumnTransformer(transformers=transformers)

pipeline = Pipeline(
    steps=[
        ("pre", preprocessor),
        ("select", SelectKBest(score_func=f_classif)),
        ("clf", LogisticRegression(max_iter=1000, solver="liblinear")),
    ]
)

param_grid = {
    "select__k": [10, 15, 20, "all"],
    "clf__C": [0.01, 0.1, 1, 3],
    "clf__class_weight": [None, "balanced", {0: 1, 1: 0.5}, {0: 1, 1: 0.25}],
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="balanced_accuracy",
    cv=10,
    n_jobs=-1,
    verbose=0,
)

grid.fit(X_train, y_train)

save_model_gzip(grid, "files/models/model.pkl.gz")

pred_train = grid.predict(X_train)
pred_test = grid.predict(X_test)

def metrics_dict(y_true, y_pred, dataset_name):
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def cm_dict(y_true, y_pred, dataset_name):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = int(cm[0, 0]) if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
        fp = int(cm[0, 1]) if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
        fn = int(cm[1, 0]) if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
        tp = int(cm[1, 1]) if cm.shape[0] > 1 and cm.shape[1] > 1 else 0

    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
        "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)},
    }


rows = [
    metrics_dict(y_train, pred_train, "train"),
    metrics_dict(y_test, pred_test, "test"),
    cm_dict(y_train, pred_train, "train"),
    cm_dict(y_test, pred_test, "test"),
]

write_metrics_jsonlines(rows, "files/output/metrics.json")