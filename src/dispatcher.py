
## import necessary packages/modules
from sklearn import linear_model
from sklearn import ensemble
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

## all the models here are tuned 

def get_models(model_type):
    """
    Function to get the models.

    Args:
        model_type (str): type of the model.
    
    Returns:
        model
    """    
    if model_type == "basic":
        MODELS = {
            "log_regg": linear_model.LogisticRegression(
                C=0.3593813663804626,
                penalty="l2",
                solver="saga",
                n_jobs=-1,
                max_iter=200,
                verbose=1
            ),

            "random_forest": ensemble.RandomForestClassifier(
                criterion="entropy",
                max_depth=5,
                n_estimators=100,
                min_samples_split=2,
                n_jobs=-1,
                verbose=1
            ),

            "xg_boost": XGBClassifier(
                min_child_weight=5,
                max_depth=4,
                learning_rate=0.05,
                gamma=0.,
                colsample_bytree=0.7,
                n_jobs=-1,
                verbosity=1
            )
        }
    
    elif model_type == "intermediate":
        MODELS = {
            "log_regg": linear_model.LogisticRegression(
                C=0.3593813663804626,
                penalty="l2",
                solver="liblinear"
            ),

            "random_forest": ensemble.RandomForestClassifier(
                criterion="entropy",
                max_depth=7,
                min_samples_split=2,
                n_estimators=400,
                n_jobs=-1,
                verbose=1
            ),

            "xg_boost": XGBClassifier(
                min_child_weight=7,
                max_depth=4,
                learning_rate=0.05,
                gamma=0.4,
                colsample_bytree=1,
                n_jobs=-1,
                verbosity=1
            )
        }

    elif model_type == "advance":
        MODELS = {
            "log_regg": linear_model.LogisticRegression(
                C=0.3593813663804626,
                penalty="l2",
                solver="lbfgs"
            ),

            "random_forest": ensemble.RandomForestClassifier(
                criterion="entropy",
                max_depth=7,
                min_samples_split=2,
                n_estimators=100
            ),

            "xg_boost": XGBClassifier(
                min_child_weight=5,
                max_depth=3,
                learning_rate=0.1,
                gamma=0.3,
                colsample_bytree=0.3
            )
        }

    return MODELS
