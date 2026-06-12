import os
import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object # Helper function to serialize components

@dataclass
class DataTransformationConfig:
    # Path where the saved preprocessor binary artifact will be dumped
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, num_features, cat_features):
        """
        Creates the isolated, reusable pipeline blocks for scaling and encoding.
        """
        try:
            # Numerical Pipeline
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Categorical Pipeline
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info("Numerical and Categorical pipelines initialized successfully.")

            # Combine pipelines into a single transformer engine
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, num_features),
                ("cat_pipeline", cat_pipeline, cat_features)
            ])

            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads train/test splits, runs them through transformation pipelines,
        and saves the preprocessor object configuration.
        """
        try:
            # 1. Load the split datasets from the incoming paths
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Successfully read train and test data components.")
            logging.info("Obtaining preprocessor configuration pipeline object...")

            # Define your feature columns (Update these strings based on your actual dataset schema)
            target_column_name = "math_score" 
            
            # Filter numerical and categorical columns automatically, excluding the target
            numerical_columns = train_df.select_dtypes(include=["int64", "float64"]).columns.drop(target_column_name, errors='ignore').tolist()
            categorical_columns = train_df.select_dtypes(include=["object"]).columns.tolist()

            # 2. Get our structured ColumnTransformer pipeline
            preprocessing_obj = self.get_data_transformer_object(
                num_features=numerical_columns, 
                cat_features=categorical_columns
            )

            # 3. Separate Independent (X) and Dependent (y) variables
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing pipeline on training and testing dataframes.")

            # 4. Transform the feature spaces
            # Fit and transform on training data calculates the metrics (mean, std, categories)
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            
            # ONLY transform on test data to prevent target data leakage!
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # 5. Concatenate features and targets back together into efficient numpy arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessor configuration binary artifact.")

            # 6. Save the preprocessor pipeline object to the artifacts folder
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return arrays and the object path to be easily ingested by model_trainer.py
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)