import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
from joblib import dump
from scipy.sparse import issparse

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")
    label_encoder_obj_file_path: str = os.path.join('artifacts', "label_encoder.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Creating data transformer object")
            text_feature_column = "contract_text"
            # --- UPDATED: Increased max_features for a richer vocabulary ---
            preprocessor = ColumnTransformer(
                transformers=[
                    ('tfidf_vectorizer', TfidfVectorizer(stop_words='english', max_features=10000), text_feature_column)
                ],
                remainder='drop'
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data loaded successfully.")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "risk_label"
            X_train_df = train_df.drop(columns=[target_column_name], axis=1)
            y_train_df = train_df[target_column_name]

            X_test_df = test_df.drop(columns=[target_column_name], axis=1)
            y_test_df = test_df[target_column_name]

            X_train_transformed = preprocessing_obj.fit_transform(X_train_df)
            X_test_transformed = preprocessing_obj.transform(X_test_df)

            if issparse(X_train_transformed):
                X_train_transformed = X_train_transformed.toarray()
            if issparse(X_test_transformed):
                X_test_transformed = X_test_transformed.toarray()

            logging.info("Encoding target variable and saving the encoder.")
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train_df)
            y_test_encoded = label_encoder.transform(y_test_df)
            
            with open(self.data_transformation_config.label_encoder_obj_file_path, "wb") as file_obj:
                dump(label_encoder, file_obj)

            logging.info("Saving preprocessing object.")
            with open(self.data_transformation_config.preprocessor_obj_file_path, "wb") as file_obj:
                dump(preprocessing_obj, file_obj)
            
            logging.info("Data transformation completed successfully.")

            return (
                X_train_transformed,
                y_train_encoded,
                X_test_transformed,
                y_test_encoded
            )
        except Exception as e:
            raise CustomException(e, sys)
