import os
import sys
import pickle
from src.exception import CustomException
from sklearn.metrics import r2_score


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
    

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Loops through a dictionary of models, handles fitting, 
    and returns an performance scorecard dictionary.
    """
    try:
        report = {}
        for model_name, model in models.items():
            # Fit/Train the model parameters
            model.fit(X_train, y_train)
            
            # Predict against unseen data
            y_test_pred = model.predict(X_test)
            
            # Calculate the evaluation metric (R-squared score)
            test_model_score = r2_score(y_test, y_test_pred)
            
            # Append to our performance scorecard map
            report[model_name] = test_model_score
            
        return report
        
    except Exception as e:
        raise CustomException(e, sys)