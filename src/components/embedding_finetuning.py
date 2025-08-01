import os
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from src.logger import logging
from src.exception import CustomException

class EmbeddingFinetuner:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.num_epochs = 1
        self.batch_size = 16
        self.model_save_path = "artifacts/finetuned_sentence_transformer"
        os.makedirs(self.model_save_path, exist_ok=True)

    def initiate_embedding_finetuning(self, train_path: str) -> str:
        try:
            logging.info("Reading training data for fine-tuning embeddings.")
            train_df = pd.read_csv(train_path)

            # Drop NaNs
            train_df.dropna(subset=["contract_text", "risk_label"], inplace=True)

            # Remove rows where contract_text or risk_label is None, 'None', or empty after strip
            train_df = train_df[
                train_df["contract_text"].apply(lambda x: isinstance(x, str) and x.strip() and x.strip().lower() != "none")
            ]
            train_df = train_df[
                train_df["risk_label"].apply(lambda x: isinstance(x, str) and x.strip() and x.strip().lower() != "none")
            ]

            logging.info(f"After cleaning, {len(train_df)} valid samples remain.")

            # Encode labels
            label_encoder = LabelEncoder()
            train_df["encoded_label"] = label_encoder.fit_transform(train_df["risk_label"])

            # Prepare training examples
            train_examples = []
            for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Preparing training examples"):
                try:
                    contract = row["contract_text"].strip()
                    label = int(row["encoded_label"])
                    train_examples.append(InputExample(texts=[contract, contract], label=label))
                except Exception as inner_e:
                    logging.warning(f"Skipping row due to error: {inner_e}")
                    continue

            if len(train_examples) == 0:
                raise ValueError("No valid training examples found after cleaning.")

            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=self.batch_size)
            train_loss = losses.CosineSimilarityLoss(self.model)

            logging.info("Starting embedding fine-tuning...")
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=self.num_epochs,
                warmup_steps=10,
                show_progress_bar=True
            )

            self.model.save(self.model_save_path)
            logging.info(f"Fine-tuned model saved at: {self.model_save_path}")
            return self.model_save_path

        except Exception as e:
            logging.error("Error during embedding fine-tuning.")
            raise CustomException(e, sys)
