# Threat Text Classification

Performing classification task on texts having threat contents.

---

#### DATASET URLS
- https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

#### PROJECT SETUP
```bash
make -f Makefile
python init.py
```

#### RUN PRE-COMMIT
```bash
pre-commit install
```

#### RUN PROJECT
```bash
python run.py
```

#### PERFORMANCE OVER VALIDATION DATA
```
Classification report :
              precision    recall  f1-score   support

           0       0.96      0.86      0.91        80
           1       0.88      0.96      0.92        80

    accuracy                           0.91       160
   macro avg       0.92      0.91      0.91       160
weighted avg       0.92      0.91      0.91       160
```

#### LOSS CHART
![Loss chart image](/outputs/loss_chart.png)

#### ACCURACY CHART
![Accuracy chart image](/outputs/accuracy_chart.png)

#### PRECISION VS RECALL CURVE
![Precision vs recall curve image](/outputs/precision_vs_recall_chart.png)

#### PRECISION-RECALL VS THRESHOLD CHART
![Precision recall vs threshold image](/outputs/precision_recall_vs_threshold_chart.png)

#### ROC CURVE
![Roc curve image](/outputs/roc_curve.png)