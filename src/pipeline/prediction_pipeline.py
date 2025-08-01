import os
import sys
import pandas as pd
from pathlib import Path
from joblib import load
import json

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.exception import CustomException
from src.logger import logging

class PredictionPipeline:
    def __init__(self):
        try:
            # Load all the necessary artifacts
            self.embedder = load(os.path.join("artifacts", "embedder.pkl"))
            self.model = load(os.path.join("artifacts", "model_lgbm.pkl"))
            self.le = load(os.path.join("artifacts", "label_encoder.pkl"))
            logging.info("Loaded embedder, model, and label encoder artifacts.")
        except Exception as e:
            raise CustomException(e, sys)

    async def get_improvement_suggestion(self, raw_contract_text: str):
        """
        Uses the Gemini API to generate a suggestion for improving a risky clause.
        """
        logging.info("Calling Gemini API for improvement suggestion.")
        try:
            # Construct a clear and specific prompt for the Gemini API
            prompt = (
                "You are an expert legal analyst. Review the following contract clause, "
                "identify the primary risks, and suggest a more balanced and safer alternative. "
                "Present your answer clearly with a 'Risk Analysis' section and a 'Suggested Rewrite' section.\n\n"
                f"Clause to review: \"{raw_contract_text}\""
            )

            # Prepare the payload for the Gemini API call
            chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
            payload = {"contents": chat_history}
            
            # The API key is handled by the environment, so it can be left empty.
            api_key = "" 
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

            # Asynchronous API call (requires an async environment like the one in Flask/Streamlit)
            # Note: For a purely synchronous script, you would use a library like 'requests'.
            # Here we simulate the fetch call structure.
            # In a real async app, you'd use a library like `aiohttp`.
            # For this context, we'll use a placeholder for the fetch logic.
            # This part will be executed by the browser's fetch in a web context.
            
            # This is a conceptual representation. The actual fetch happens in the web app.
            # We will return the prompt and let the frontend handle the call.
            # In a real backend-only system, you would use `requests` or `httpx`.
            
            # For our pipeline, we will just return the prompt for now.
            # The actual API call will be made in the Flask app.
            return prompt

        except Exception as e:
            logging.error(f"Error in get_improvement_suggestion: {e}")
            return "Could not generate a suggestion due to an error."


    def predict(self, raw_contract_text: str):
        """
        Predicts the risk level and prepares a suggestion prompt if the risk is high or medium.
        """
        try:
            # Embed the new text
            embeddings = self.embedder.encode([raw_contract_text])

            # Predict risk class index
            pred_idx = self.model.predict(embeddings)
            pred_label = self.le.inverse_transform(pred_idx)[0]

            # Prepare the final result dictionary
            result = {"risk_level": pred_label, "suggestion_prompt": None}

            # If the risk is not low, prepare the prompt for Gemini
            if pred_label in ['high', 'medium']:
                logging.info(f"High/Medium risk detected ({pred_label}). Preparing suggestion prompt.")
                result["suggestion_prompt"] = (
                    "You are an expert legal analyst. Review the following contract clause, "
                    "identify the primary risks, and suggest a more balanced and safer alternative. "
                    "Present your answer clearly with a 'Risk Analysis' section and a 'Suggested Rewrite' section.\n\n"
                    f"Clause to review: \"{raw_contract_text}\""
                )
            else:
                 result["suggestion_prompt"] = "No improvement suggestions needed for a low-risk clause."


            return result

        except Exception as e:
            raise CustomException(e, sys)
