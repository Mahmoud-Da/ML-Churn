import torch.nn as nn
from src.config import INPUT_FEATURES, HIDDEN_UNITS_1, HIDDEN_UNITS_2, OUTPUT_UNITS


class ChurnPredictorMLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) Neural Network.
    It takes customer features, passes them through hidden layers, 
    and outputs a probability (0 to 1) of the customer churning.
    """

    def __init__(self):
        super(ChurnPredictorMLP, self).__init__()

        # Layer 1: Input to Hidden Layer 1
        self.layer1 = nn.Linear(INPUT_FEATURES, HIDDEN_UNITS_1)
        self.relu1 = nn.ReLU()  # Activation function adds non-linearity

        # Layer 2: Hidden Layer 1 to Hidden Layer 2
        self.layer2 = nn.Linear(HIDDEN_UNITS_1, HIDDEN_UNITS_2)
        self.relu2 = nn.ReLU()

        # Layer 3: Hidden Layer 2 to Output
        self.output_layer = nn.Linear(HIDDEN_UNITS_2, OUTPUT_UNITS)
        # Sigmoid squashes the final output to a number between 0 and 1 (a probability)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Defines how data flows through the network."""
        x = self.layer1(x)
        x = self.relu1(x)

        x = self.layer2(x)
        x = self.relu2(x)

        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x
