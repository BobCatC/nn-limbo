import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    tp = np.sum(np.logical_and(prediction, ground_truth))
    tn = np.sum(np.logical_not(np.logical_or(prediction, ground_truth)))
    fn = np.sum(np.logical_and(np.logical_not(prediction), ground_truth))
    fp = np.sum(np.logical_and(prediction, np.logical_not(ground_truth)))

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fp) != 0 else 0
    f1 = 2 * recall * precision / (precision + recall) if (precision + recall) != 0 else 0
    accuracy = (tp + tn) / prediction.shape[0]

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    return np.sum(prediction == ground_truth) / prediction.shape[0]
