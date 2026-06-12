import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    """Defines the path where the final trained model binary will be saved."""
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Splits arrays into features/targets, trains multiple algorithms,
        evaluates performance, and serializes the top-performing model.
        """
        try:
            logging.info("Splitting training and test input data matrices.")
            
            # Since the target column was appended to the end in data_transformation.py:
            # [:, :-1] selects all columns except the last one (Features)
            # [:, -1] selects only the last column (Target)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Dictionary of algorithms to benchmark
            models = {
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            param_grids = {
                # --- Linear Models ---
                "Linear Regression": {}, # No hyperparameters to tune

                
                # --- Tree & Distance-Based Models ---
                "Decision Tree": {
                    'criterion': ['squared_error', 'absolute_error'],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                # --- Ensemble Models ---
                "Random Forest": {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },

                "Gradient Boosting": {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'max_depth': [3, 5, 8]
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 150, 200],
                    'learning_rate': [0.001, 0.01, 0.05, 0.1, 1.0],
                    'loss': ['linear', 'square', 'exponential']
                },
                
                # --- Advanced Boosting Frameworks ---
                "XGBRegressor": {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 10],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                },

            }

            logging.info("Initiating model evaluation loop across algorithm dictionary.")
            
            # evaluate_models is a helper function from src.utils that handles the fitting and scoring
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, 
                X_test=X_test, y_test=y_test, 
                models=models,param_grids=param_grids
            )
            
            # Find the highest R2 score from the returned evaluation dictionary
            best_model_score = max(sorted(model_report.values()))

            # Extract the name of the highest scoring model
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # Quality Guardrail: Reject training if the best model performance is too low
            if best_model_score < 0.6:
                raise CustomException("No suitable model configuration found with an R2 score above 0.6", sys)
            
            logging.info(f"Optimal model isolated: {best_model_name} with R2 Score: {best_model_score}")

            # Serialize and save the winning model to the artifacts directory
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Return the final evaluation metric score
            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)