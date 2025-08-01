import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw.csv")
    # Pointing to the correct CSV file
    cuad_data_path: str = os.path.join('data', "master_clauses.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def map_risk_levels(self):
        """
        Defines the mapping from CUAD categories to risk levels.
        UPDATED: Keys now exactly match the column headers from the provided file sample.
        """
        risk_mapping = {
            # High Risk Categories
            'Uncapped Liability': 'high', 'Most Favored Nation': 'high', 'Non-Compete': 'high',
            'Ip Ownership Assignment': 'high', 'Irrevocable Or Perpetual License': 'high',
            'Liquidated Damages': 'high', 'Termination For Convenience': 'high',
            'Exclusivity': 'high', 'Covenant Not To Sue': 'high',

            # Medium Risk Categories
            'Change Of Control': 'medium', 'Anti-Assignment': 'medium', 'No-Solicit Of Employees': 'medium',
            'No-Solicit Of Customers': 'medium', 'Post-Termination Services': 'medium',
            'Audit Rights': 'medium', 'Cap On Liability': 'medium', 'Source Code Escrow': 'medium',
            'Rofr/Rofo/Rofn': 'medium', # FIX: Using abbreviation
            'Minimum Commitment': 'medium', 'Volume Restriction': 'medium', 
            'Price Restrictions': 'medium', # FIX: Plural
            'Revenue/Profit Sharing': 'medium', 'Joint Ip Ownership': 'medium',

            # Low Risk Categories
            'Governing Law': 'low', 'Agreement Date': 'low', 'Effective Date': 'low',
            'Expiration Date': 'low', 'Renewal Term': 'low', 
            'Notice Period To Terminate Renewal': 'low', # FIX: Added "Period To"
            'Parties': 'low', 'Document Name': 'low', 'Warranty Duration': 'low',
            'Insurance': 'low', 'Third Party Beneficiary': 'low', 'Non-Disparagement': 'low',
            'Competitive Restriction Exception': 'low', 'License Grant': 'low',
            'Non-Transferable License': 'low', 
            'Affiliate License-Licensor': 'low', # FIX: Hyphen only
            'Affiliate License-Licensee': 'low', # FIX: Hyphen only
            'Unlimited/All-You-Can-Eat-License': 'low' # FIX: Hyphens
        }
        # The script will now also try to match case-insensitively for robustness.
        return risk_mapping

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion using CUAD dataset")
        try:
            df = pd.read_csv(self.ingestion_config.cuad_data_path)
            logging.info(f"Successfully loaded {self.ingestion_config.cuad_data_path}")

            # Make column matching case-insensitive for robustness
            df.columns = df.columns.str.strip()
            
            risk_mapping = self.map_risk_levels()
            
            # Create a case-insensitive map for matching
            column_map = {col.lower(): col for col in df.columns}
            mapped_risk_keys = {}
            missing_columns = []

            for key in risk_mapping.keys():
                if key.lower() in column_map:
                    mapped_risk_keys[column_map[key.lower()]] = risk_mapping[key]
                else:
                    missing_columns.append(key)

            if missing_columns:
                logging.error("The loaded CSV file is still missing expected columns after case-insensitive matching.")
                logging.error(f"--- MISSING CATEGORIES: {missing_columns}")
                logging.error(f"--- COLUMNS FOUND IN FILE: {df.columns.tolist()}")
                raise ValueError("Input file does not have the expected columns. Check logs for details.")
            
            logging.info("Data validation successful. All required columns are present.")

            # Melt the dataframe using the correctly identified column names
            df_melted = df.melt(id_vars=['Filename'], value_vars=list(mapped_risk_keys.keys()), var_name='category', value_name='contract_text')

            df_melted.dropna(subset=['contract_text'], inplace=True)
            # Use the original risk_mapping to get the risk label
            df_melted['risk_label'] = df_melted['category'].map(mapped_risk_keys)
            df_final = df_melted.dropna(subset=['risk_label'])
            df_final = df_final[['contract_text', 'risk_label']]

            logging.info(f"Processed CUAD data. Found {len(df_final)} labeled clauses.")
            if len(df_final) == 0:
                raise Exception("No data was processed from the CUAD file. Check paths and file content.")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df_final.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Initiating train-test split")
            train_set, test_set = train_test_split(df_final, test_size=0.2, random_state=42, stratify=df_final['risk_label'])
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the CUAD data is complete.")
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        
        except FileNotFoundError:
            logging.error(f"FATAL: The CUAD dataset was not found at {self.ingestion_config.cuad_data_path}")
            logging.error("Please ensure 'master_clauses.csv' is in the 'data' folder.")
            raise CustomException(f"CUAD dataset not found at {self.ingestion_config.cuad_data_path}", sys)
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()
