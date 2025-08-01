# src/components/model_training.py
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from joblib import dump

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import randint, uniform

# Add project root to PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model_lgbm.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        """
        1) Runs RandomizedSearchCV (20 trials) to find good hyperparameters.
        2) Retrains a final LightGBM model with early stopping.
        3) Saves the final model to disk.
        """
        try:
            logging.info("Starting RandomizedSearchCV over LightGBM hyperparameters")

            # Compute sample weights to handle imbalance
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

            # Base estimator
            base_est = LGBMClassifier(
                objective="multiclass",
                n_estimators=300,
                learning_rate=0.05,
                class_weight="balanced",
                verbose=-1
            )

            # Hyperparameter distributions
            param_dist = {
                "num_leaves": randint(20, 200),
                "max_depth": randint(3, 15),
                "min_child_samples": randint(5, 100),
                "subsample": uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.6, 0.4),
                "reg_alpha": uniform(0, 1),
                "reg_lambda": uniform(1, 10)
            }

            search = RandomizedSearchCV(
                estimator=base_est,
                param_distributions=param_dist,
                n_iter=20,        # only 20 random trials
                cv=3,
                n_jobs=-1,
                verbose=2,
                random_state=42
            )

            # Run the search
            search.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_test, y_test)],
                eval_metric="multi_logloss",
                callbacks=[early_stopping(stopping_rounds=20, verbose=False),
                           log_evaluation(period=0)]
            )

            best_params = search.best_params_
            logging.info(f"RandomizedSearchCV complete. Best params: {best_params}")
            print(f"Best params: {best_params}")

            # 2) Retrain final model with early stopping
            logging.info("Retraining final model with early stopping using best parameters")
            final_model = LGBMClassifier(
                objective="multiclass",
                class_weight="balanced",
                **best_params,
                n_estimators=500,       # give more room; early stopping will prune
                learning_rate=best_params.get("learning_rate", 0.05),
                verbose=-1
            )
            final_model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_test, y_test)],
                callbacks=[early_stopping(stopping_rounds=20), log_evaluation(period=0)]
            )

            # 3) Evaluate & save
            acc = final_model.score(X_test, y_test)
            logging.info(f"Final LightGBM test accuracy: {acc:.4f}")
            print(f"Final test accuracy: {acc:.4f}")

            os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)
            dump(final_model, self.config.trained_model_file_path)
            logging.info(f"Model saved to {self.config.trained_model_file_path}")

            return self.config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)
