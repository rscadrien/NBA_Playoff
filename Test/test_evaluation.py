import numpy as np
import pytest
from unittest import mock
from Training.evaluation import evaluation

@pytest.fixture
def sample_data():
    y_test = np.array([
        [0, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 0, 0, 1],
        [1, 0, 1, 1]
    ])
    y_pred = np.array([
        [0, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 1]
    ])
    y_prob = np.array([
        [0.1, 0.8, 0.9, 0.9],
        [0.9, 0.7, 0.9, 0.95],
        [0.2, 0.6, 0.2, 0.9],
        [0.85, 0.2, 0.8, 0.97]
    ])
    class_names = ['Champion', 'Finalist', 'Conf_Finalist', 'Conf_SemiFinalist']
    return y_test, y_pred, y_prob, class_names

@mock.patch("Evaluation.evaluation.ConfusionMatrixDisplay.from_predictions")
@mock.patch("Evaluation.evaluation.plt.show")
def test_evaluate_model_runs(mock_show, mock_conf_matrix, sample_data):
    y_test, y_pred, y_prob, class_names = sample_data

    # Run evaluation
    evaluation(y_test, y_pred, y_prob, class_names)

    # Check confusion matrix is displayed for each class
    assert mock_conf_matrix.call_count == len(class_names)
    assert mock_show.call_count == len(class_names)