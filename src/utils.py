import os
import sys
import pickle
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from src.logger import logging 

def save_object(file_path, obj):
    """
    Serializes a Python object and saves it as a binary file (.pkl) at the specified path.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param_grids):
    """
    Loops through a dictionary of models, tunes them safely using RandomizedSearchCV,
    and returns a performance scorecard dictionary using the best parameters.
    """
    try:
        report = {}
        
        for model_name, model in models.items():
            params = param_grids.get(model_name, {})
            
            # SAFEGUARD: If param grid is empty, bypass tuning loop completely
            if not params:
                logging.info(f"Bypassing hyperparameter tuning for baseline model: {model_name}")
                model.fit(X_train, y_train)
                best_model = model
            else:
                logging.info(f"Initiating RandomizedSearchCV optimization for: {model_name}")
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=params,
                    n_iter=5, # Reduced slightly to keep evaluation fast across all 13 models
                    cv=3,
                    scoring="r2",
                    n_jobs=-1,
                    random_state=42,
                    error_score='raise'
                )
                search.fit(X_train, y_train)
                
                # Assign the pre-trained, optimized model asset
                best_model = search.best_estimator_
            
            # Predict and score against unseen test data (No redundant fitting!)
            y_test_pred = best_model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            
            # Track the score of the tuned model configuration
            report[model_name] = test_model_score
            logging.info(f"Model: {model_name} | Evaluation R2 Score: {test_model_score}")
            
        return report
        
    except Exception as e:
        raise CustomException(e, sys)