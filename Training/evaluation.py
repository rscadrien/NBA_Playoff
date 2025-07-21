import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             brier_score_loss, ConfusionMatrixDisplay)

def evaluate_model(y_test, y_pred, y_prob, class_names):
    for i, label_name in enumerate(class_names):
        print(f"--- {label_name} ---")
        acc = accuracy_score(y_test[:, i], y_pred[:, i])
        prec = precision_score(y_test[:, i], y_pred[:, i])
        rec = recall_score(y_test[:, i], y_pred[:, i])
        f1 = f1_score(y_test[:, i], y_pred[:, i])
        brier = brier_score_loss(y_test[:, i], y_prob[:, i])

        print(f"Accuracy: {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall: {rec:.3f}")
        print(f"F1: {f1:.3f}")
        print(f"Brier Score: {brier:.3f}")

        ConfusionMatrixDisplay.from_predictions(y_test[:, i], y_pred[:, i], normalize="true", values_format=".0%")
        plt.show()