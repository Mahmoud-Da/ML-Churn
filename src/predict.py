import torch
import numpy as np
from src.model import ChurnPredictorMLP
from src.config import MODEL_PATH, INPUT_FEATURES


def predict_single_customer(features):
    """
    Takes a list of 10 features for a single customer and predicts churn.
    """
    # 1. Load the architecture
    model = ChurnPredictorMLP()

    # 2. Load the trained weights into the architecture
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()  # Set model to evaluation mode (turns off dropout/batchnorm if used)
    except FileNotFoundError:
        print("Error: Model not found. Run 'python src/train.py' first!")
        return

    # 3. Prepare data (convert to tensor)
    # Note: In a real app, you MUST scale this data using the scaler from data_prep.py
    input_tensor = torch.tensor([features], dtype=torch.float32)

    # 4. Predict
    with torch.no_grad():  # We don't need gradients for predicting, saves memory
        prediction = model(input_tensor)
        probability = prediction.item()

    print(f"\nCustomer Features: {features}")
    print(f"Churn Probability: {probability:.2%}")
    if probability > 0.5:
        print("Conclusion: HIGH RISK of Churn. Offer a discount!")
    else:
        print("Conclusion: LOW RISK of Churn. Customer is happy.")


if __name__ == "__main__":
    # Create some dummy data for a single customer (10 features)
    dummy_customer = np.random.uniform(-1, 1, INPUT_FEATURES).tolist()
    predict_single_customer(dummy_customer)
