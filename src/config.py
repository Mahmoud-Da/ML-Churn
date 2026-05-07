import os

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "churn_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pth")

# --- HYPERPARAMETERS ---
INPUT_FEATURES = 10     # Number of columns/features in our data
HIDDEN_UNITS_1 = 64     # Neurons in the first hidden layer
HIDDEN_UNITS_2 = 32     # Neurons in the second hidden layer
OUTPUT_UNITS = 1        # Binary classification (0 or 1)

LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50
