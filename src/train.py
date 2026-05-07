import torch
import torch.nn as nn
import torch.optim as optim
from src.model import ChurnPredictorMLP
from src.data_prep import get_dataloaders
from src.config import LEARNING_RATE, EPOCHS, MODEL_PATH
import os


def train_model():
    print("--- Starting Machine Learning Pipeline ---")

    # 1. Load Data
    train_loader, test_loader, _ = get_dataloaders()

    # 2. Initialize Model
    model = ChurnPredictorMLP()

    # 3. Define Loss Function and Optimizer
    # BCELoss (Binary Cross Entropy) is standard for predicting 0 or 1
    criterion = nn.BCELoss()
    # Adam Optimizer updates the network's weights based on the loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting Training Loop...")
    for epoch in range(EPOCHS):
        model.train()  # Set model to training mode
        current_loss = 0.0

        # Loop through batches of data
        for inputs, labels in train_loader:
            # Step A: Clear old gradients
            optimizer.zero_grad()

            # Step B: Forward Pass (Make predictions)
            predictions = model(inputs)

            # Step C: Calculate Loss (How wrong were the predictions?)
            loss = criterion(predictions, labels)

            # Step D: Backward Pass (Calculate gradients - how to fix the weights)
            loss.backward()

            # Step E: Optimize (Update the weights)
            optimizer.step()

            current_loss += loss.item()

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} | Loss: {current_loss/len(train_loader):.4f}")

    # Save the trained model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nTraining Complete! Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
