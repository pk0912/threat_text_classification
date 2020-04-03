"""
Python file to evaluate the built model
"""

import os
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

from utils.visualisation import line_chart_with_x, line_chart_with_x_y


def plot_loss_chart(history, save_path):
    loss_chart_data = [
        {"value": history.history["loss"], "label": "loss", "style": "-"},
        {"value": history.history["val_loss"], "label": "val_loss", "style": "-"},
    ]
    line_chart_with_x(
        loss_chart_data,
        os.path.join(save_path, "loss_chart.png"),
        "Epochs -->",
        "Loss -->",
    )


def plot_accuracy_chart(history, save_path):
    acc_chart_data = [
        {"value": history.history["accuracy"], "label": "acc", "style": "-"},
        {"value": history.history["val_accuracy"], "label": "val_acc", "style": "-"},
    ]
    line_chart_with_x(
        acc_chart_data,
        os.path.join(save_path, "accuracy_chart.png"),
        "Epochs -->",
        "Loss -->",
    )


def plot_precision_recall_chart(y_true, y_pred_probs, save_path):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_probs)
    prec_rec_thresh_data = [
        {"X": thresholds, "y": precisions[:-1], "label": "Precision", "style": "b--"},
        {"X": thresholds, "y": recalls[:-1], "label": "Recall", "style": "g-"},
    ]
    prec_rec_data = [
        {"X": recalls, "y": precisions, "label": "precision_recall", "style": "b-"}
    ]
    line_chart_with_x_y(
        prec_rec_thresh_data,
        os.path.join(save_path, "precision_recall_vs_threshold_chart.png"),
        "Threshold",
    )
    line_chart_with_x_y(
        prec_rec_data,
        os.path.join(save_path, "precision_vs_recall_chart.png"),
        "Recall",
        "Precision",
    )


def plot_roc_curve(y_true, y_pred_probs, save_path):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_data = [
        {"X": fpr, "y": tpr, "label": "fpr_tpr", "style": "-"},
        {"X": [0, 1], "y": [0, 1], "label": "middle_line", "style": "k--"},
    ]
    line_chart_with_x_y(
        roc_data,
        os.path.join(save_path, "roc_curve.png"),
        "False Positive Rate",
        "True Positive Rate (Recall)",
    )


def save_performance(model, X, y, save_path, file_suffix):
    probabilities = model.predict(X)
    y_pred = np.where(probabilities > 0.5, 1, 0)
    file_name = "evaluation_{}.txt".format(file_suffix)
    with open(os.path.join(save_path, file_name), "w") as f:
        f.write("Shape of predicted values : {}".format(y_pred.shape))
        f.write("\n")
        f.write("Shape of target values : {}".format(y.shape))
        f.write("\n")
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        f.write("True neg : {}".format(tn))
        f.write("\n")
        f.write("False pos : {}".format(fp))
        f.write("\n")
        f.write("False neg : {}".format(fn))
        f.write("\n")
        f.write("True pos : {}".format(tp))
        f.write("\n")
        f.write("ROC : {}".format(roc_auc_score(y, y_pred)))
        f.write("\n")
        f.write("Precision : {}".format(precision_score(y, y_pred)))
        f.write("\n")
        f.write("Accuracy : {}".format(accuracy_score(y, y_pred)))
        f.write("\n")
        f.write("Recall : {}".format(recall_score(y, y_pred)))
        f.write("\n")
        f.write("F1 : {}".format(f1_score(y, y_pred)))
        f.write("\n")
        f.write("Classification report : ")
        f.write("\n")
        f.write(classification_report(y, y_pred))
        f.write("\n")


def evaluate(model, history, X, y, save_path, file_suffix="val"):
    plot_loss_chart(history, save_path)
    plot_accuracy_chart(history, save_path)
    y_pred_probs = model.predict(X)
    plot_precision_recall_chart(y, y_pred_probs, save_path)
    plot_roc_curve(y, y_pred_probs, save_path)
    save_performance(model, X, y, save_path, file_suffix)
