import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

# Load the saved prediction datasets
SAVE_PATH = "output/_dataset_results"  # use your actual SAVE_PATH here
MODEL = "log_regg"  # or whatever your model name is

train_preds_path = os.path.join(SAVE_PATH, f'train_preds_{MODEL}.pkl')
test_preds_path = os.path.join(SAVE_PATH, f'test_preds_{MODEL}.pkl')

train_df = pd.read_pickle(train_preds_path)
test_df = pd.read_pickle(test_preds_path)

# True targets and predicted probabilities
y_train = train_df['target']
y_test = test_df['target']
preds_train = train_df['pred_' + MODEL]
preds_test = test_df['pred_' + MODEL]

# ROC curve and AUC for train
fpr_train, tpr_train, _ = roc_curve(y_train, preds_train)
roc_auc_train = auc(fpr_train, tpr_train)

# ROC curve and AUC for test
fpr_test, tpr_test, _ = roc_curve(y_test, preds_test)
roc_auc_test = auc(fpr_test, tpr_test)

plt.figure(figsize=(10,6))

# Plot ROC curves
plt.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Train ROC curve (area = {roc_auc_train:.2f})')
plt.plot(fpr_test, tpr_test, color='red', lw=2, label=f'Test ROC curve (area = {roc_auc_test:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Train and Test sets')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Histograms of features in train data
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.hist(train_df['angle'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Angle (Train)')
plt.xlabel('Angle')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.hist(train_df['distance'], bins=30, color='salmon', edgecolor='black')
plt.title('Distribution of Distance (Train)')
plt.xlabel('Distance')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Predicted probability distribution
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.hist(preds_train, bins=30, color='green', alpha=0.7, edgecolor='black')
plt.title('Predicted Probabilities (Train)')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.hist(preds_test, bins=30, color='orange', alpha=0.7, edgecolor='black')
plt.title('Predicted Probabilities (Test)')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
