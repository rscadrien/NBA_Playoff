import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score)
from sklearn.model_selection import cross_val_predict

def evaluate_model(model, X, Y, class_names):
    # Get cross-validated predictions
    Y_pred = cross_val_predict(model, X, Y, cv=5)
    # Evaluate each label/classifier
    n_labels = Y.shape[1]
    for i in range(n_labels):
        acc = accuracy_score(Y.iloc[:, i], Y_pred[:, i])
        prec = precision_score(Y.iloc[:, i], Y_pred[:, i], zero_division=0)
        rec = recall_score(Y.iloc[:, i], Y_pred[:, i], zero_division=0)
        print(f"Classifier {class_names[i]} — Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}")