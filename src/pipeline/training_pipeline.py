import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

class TrainingPipeline:
    def run(self):
        """
        Executes the full training pipeline.
        """
        logging.info("========================================")
        logging.info("Starting the Training Pipeline")
        logging.info("========================================")
        try:
            # --- Data Ingestion ---
            logging.info("--- Starting Data Ingestion ---")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            logging.info("--- Data Ingestion Finished ---")

            # --- Data Transformation ---
            logging.info("--- Starting Data Transformation ---")
            data_transformation = DataTransformation()
            # --- UPDATED: Capture separate X and y arrays ---
            X_train, y_train, X_test, y_test = data_transformation.initiate_data_transformation(
                train_path=train_data_path, test_path=test_data_path
            )
            logging.info("--- Data Transformation Finished ---")

            # --- Model Training ---
            logging.info("--- Starting Model Training ---")
            model_trainer = ModelTrainer()
            # --- UPDATED: Pass separate X and y arrays ---
            model_path = model_trainer.initiate_model_training(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
            )
            logging.info(f"Model Training complete. Model saved at: {model_path}")
            logging.info("--- Model Training Finished ---")
            
            logging.info("========================================")
            logging.info("Training Pipeline has run successfully!")
            logging.info("========================================")

        except Exception as e:
            logging.error("An error occurred during the training pipeline execution.")
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
