import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import DATA_PATH, INPUT_FEATURES, BATCH_SIZE


def generate_synthetic_data():
    """Generates fake customer data so the project runs out-of-the-box."""
    print("Generating synthetic customer churn data...")
    X, y = make_classification(
        n_samples=1000,
        n_features=INPUT_FEATURES,
        n_informative=8,
        random_state=42
    )

    # Save to CSV to mimic a real-world scenario
    df = pd.DataFrame(
        X, columns=[f"feature_{i}" for i in range(INPUT_FEATURES)])
    df["churn"] = y
    df.to_csv(DATA_PATH, index=False)
    return df


class ChurnDataset(Dataset):
    """
    PyTorch Dataset class. It tells PyTorch how to load and measure the length of our data.
    """

    def __init__(self, X, y):
        # Convert data to PyTorch Tensors (the data structure PyTorch uses for math)
        self.X = torch.tensor(X, dtype=torch.float32)
        # Reshape y from [batch_size] to [batch_size, 1] for our network
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders():
    """Prepares data, scales it, and creates PyTorch DataLoaders."""
    generate_synthetic_data()
    df = pd.read_csv(DATA_PATH)

    X = df.drop("churn", axis=1).values
    y = df["churn"].values

    # Split data: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Standardize data (Neural networks perform better when data is scaled between -1 and 1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create Datasets
    train_dataset = ChurnDataset(X_train, y_train)
    test_dataset = ChurnDataset(X_test, y_test)

    # Create DataLoaders (These feed data to the model in batches)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, scaler
