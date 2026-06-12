import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    """Defines the paths where the ingestion artifacts will be saved."""
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')


class DataIngestion:
    def __init__(self):
        # Automatically loads the paths we defined in the dataclass above
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            # 1. Read the dataset (Replace this with a database cursor or API call if needed)
            df = pd.read_csv('notebooks/data/stud.csv') 
            logging.info('Successfully read the dataset into a pandas DataFrame')

            # 2. Make sure the artifacts/ folder exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # 3. Save the completely raw data asset
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # 4. Perform the Train-Test Split
            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # 5. Save the train and test subsets to their respective paths
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data and partitioning is completed successfully")

            # Return the paths so the NEXT component (Data Transformation) knows exactly where to look
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    try:
        # Step 1: Run Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        print(f"✅ Ingestion Complete!\n   Train Path: {train_path}\n   Test Path: {test_path}\n")
        
        # Step 2: Run Data Transformation
        # Importing locally inside the block avoids circular import dependencies
        from src.components.data_transformation import DataTransformation
        
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_path=train_path, 
            test_path=test_path
        )
        print(f"✅ Transformation Complete!\n   Preprocessor saved at: {preprocessor_path}")
        print(f"   Train Matrix Shape: {train_arr.shape}")
        print(f"   Test Matrix Shape: {test_arr.shape}\n")
        
        # Step 3: Run Model Training
        from src.components.model_trainer import ModelTrainer
        
        model_trainer = ModelTrainer()
        performance_score = model_trainer.initiate_model_trainer(
            train_array=train_arr, 
            test_array=test_arr
        )
        print(f"✅ Model Training Complete!\n   Best Model Performance ($R^2$ Score): {performance_score}\n")

    except Exception as e:
        print(f"❌ Pipeline Check Failed!")
        print(e)