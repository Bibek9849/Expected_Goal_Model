'''

Python module for training models.
'''

## import necessary packages/modules
import os
import numpy as np
import pandas as pd
import joblib
from sklearn import preprocessing
from sklearn import metrics

from . import dispatcher, scaling
from dotenv import load_dotenv

load_dotenv(override=True)  # take environment variables from .env.

## get values from script file
TYPE = os.environ.get("TYPE")
TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
SAVE_PATH = os.environ.get("SAVE_PATH")
MODEL = os.environ.get("MODEL")
SCALE = os.environ.get("SCALE_TYPE")

print(TYPE)
print("fdgfdgfdgfdgdfgdg")

if __name__ == '__main__':
    ## read in the datasets
    print(TRAINING_DATA)
    train_df = pd.read_pickle(TRAINING_DATA)
    print(train_df.columns)
    test_df = pd.read_pickle(TEST_DATA)
    print("Available columns in train_df:", train_df.columns.tolist())

    ## select only numeric columns for training depending on TYPE and MODEL
    if MODEL == "log_regg":
        if TYPE == "advance":
            cols = [
                "angle", "distance", "player_in_between", "goal_keeper_angle"
            ]
        else:
            cols = [
                "angle", "distance"
            ]

        x_train = train_df[cols]
        x_test = test_df[cols]

        ## scale the values from train and test dataframe
        scale_1 = scaling.Scale(
            df=x_train,
            scale_type=SCALE,
            cols=cols
        )
        scale_2 = scaling.Scale(
            df=x_test,
            scale_type=SCALE,
            cols=cols
        )
        x_train = scale_1.fit_transform()
        x_test = scale_2.fit_transform()
    else:
        x_train = train_df.drop(["target"], axis=1)
        x_test = test_df.drop(["target"], axis=1)

    ## fetch target values for train and test dataframe
    y_train = train_df['target'].values
    y_test = test_df['target'].values

    ## train the model
    MODELS = dispatcher.get_models(TYPE)
    clf = MODELS[MODEL]
    clf.fit(x_train, y_train)

    ## predict values for train and test set
    preds_train = clf.predict_proba(x_train)[:, 1]
    preds_test = clf.predict_proba(x_test)[:, 1]

    ## add predictions directly to train_df and test_df
    train_df['pred_' + MODEL] = preds_train
    test_df['pred_' + MODEL] = preds_test

    ## calculate auc-roc score
    roc_train = metrics.roc_auc_score(y_train, preds_train)
    roc_test = metrics.roc_auc_score(y_test, preds_test)

    ## print scores
    print("*** For Train Data ***")
    print(f"ROC-AUC Score: {roc_train}\n")
    print(f"*** For Test Data ***")
    print(f"ROC-AUC Score: {roc_test}")

    ## check for directory and create if doesn't exist
   ## check for directory and create if doesn't exist
if not os.path.isdir(SAVE_PATH):
    os.makedirs(SAVE_PATH, exist_ok=True)

    ## save the datasets with predictions
    train_df.to_pickle(os.path.join(SAVE_PATH, f'train_preds_{MODEL}.pkl'))
    test_df.to_pickle(os.path.join(SAVE_PATH, f'test_preds_{MODEL}.pkl'))

    if not os.path.isdir(f"models/{TYPE}_models"):
        os.mkdir(f"models/{TYPE}_models")

    ## save the model
    joblib.dump(clf, f'models/{TYPE}_models/{MODEL}.pkl')
