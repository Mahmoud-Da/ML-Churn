# 🤖 PyTorch Customer Churn Predictor

This is a professional, containerized Machine Learning pipeline built with Python, PyTorch, Pipenv, and Docker.

## 🧠 What does this project do?

This project simulates a **Binary Classification** problem. It generates data representing customer behavior, trains a Multi-Layer Perceptron (Neural Network) to learn patterns in that data, and predicts whether a new customer is likely to "Churn" (leave the company) or stay.

## 🛠️ Tech Stack

- **PyTorch:** Used to build and train the neural network.
- **Scikit-Learn:** Used for data generation and scaling.
- **Pipenv:** Modern Python dependency management.
- **Docker:** Containerization to ensure the code runs identically on any machine.

## 🚀 How to Run Locally (Using Pipenv)

1. **Install dependencies:**

   ```bash
   pipenv install
   ```

2. **Activate the virtual environment:**
   ```bash
   pipenv shell
   ```
3. **Train the Model:** (This will generate data, train the network, and save the model)
   ```bash
   python src/train.py
   ```
4. **Make a Prediction:** (Simulate a real-time prediction on a new customer)
   ```bash
   python src/predict.py
   ```

## 🐳 How to Run using Docker (Recommended)

Don't want to install Python dependencies? Run it entirely inside a container!

1. **Build the Docker Image:**
   ```bash
   docker build -t churn-ml-project .
   ```
2. **Run the Training Pipeline:**

   ```bash
   docker run --rm -v $(pwd)/models:/app/models churn-ml-project
   ```

   _(Note: The `-v` flag maps the container's output folder to your local machine, so the saved model appears on your computer!)_

3. **Run a Prediction inside Docker:**
   ```bash
   docker run --rm -v $(pwd)/models:/app/models churn-ml-project python src/predict.py
   ```

## 📂 Architecture Overview

- **`config.py`**: A centralized place for hyperparameters. Change `EPOCHS` or `LEARNING_RATE` here.
- **`data_prep.py`**: Handles loading data and turning it into PyTorch `DataLoaders`.
- **`model.py`**: The blueprint for the Neural Network.
- **`train.py`**: The training loop (Forward pass -> Calculate Loss -> Backward pass -> Optimize).
- **`predict.py`**: Loads the `.pth` weights and makes inferences on new data.
