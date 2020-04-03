"""
Python file containing different methods for different kind of visualisation.
"""

import matplotlib.pyplot as plt


def line_chart_with_x(val_lbl_list, save_path, x_lbl="", y_lbl=""):
    plt.figure(figsize=(10, 7))
    for data in val_lbl_list:
        plt.plot(data["value"], data["style"], label=data["label"])
    plt.legend(fontsize=12)
    plt.grid()
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.savefig(save_path)


def line_chart_with_x_y(val_lbl_list, save_path, x_lbl="", y_lbl=""):
    plt.figure(figsize=(10, 7))
    for data in val_lbl_list:
        plt.plot(data["X"], data["y"], data["style"], label=data["label"])
    plt.legend(fontsize=12)
    plt.grid()
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.savefig(save_path)
