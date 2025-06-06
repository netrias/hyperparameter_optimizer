# src/loss_functions/mean_squared_error.py

import numpy as np
import json
from .base import BaseLossFunction

class MeanSquaredErrorLoss(BaseLossFunction):
    """
    Mean Squared Error (MSE) Loss with customizable penalties for over- and under-predictions.
    """

    def __init__(self, weight_over=1.0, weight_under=1.0):
        """
        Initialize the MSE loss with weights for over-predictions and under-predictions.

        Args:
            weight_over (float, optional): Weight for penalizing over-predictions. Defaults to 1.0.
            weight_under (float, optional): Weight for penalizing under-predictions. Defaults to 1.0.
        """
        super().__init__(name="MeanSquaredErrorLoss")
        self.parameters = {
            "weight_over": weight_over,  # Weight for over-predictions (y_pred > y_true)
            "weight_under": weight_under # Weight for under-predictions (y_pred < y_true)
        }

    def compute(self, y_true, y_pred):
        """
        Compute the weighted mean squared error loss.

        Args:
            y_true (array-like): True continuous values.
            y_pred (array-like): Predicted continuous values.

        Returns:
            float: Computed loss value.
        """
        # Ensure y_true and y_pred are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Compute squared error
        error = y_pred - y_true
        squared_error = error ** 2

        # Apply weights based on over- or under-predictions
        weight_mask = np.where(error > 0, self.parameters["weight_over"], self.parameters["weight_under"])
        weighted_loss = weight_mask * squared_error

        return np.mean(weighted_loss)

    @classmethod
    def deserialize(cls, data: str):
        """
        Deserialize the MeanSquaredErrorLoss from a JSON string.

        Args:
            data (str): Serialized JSON string representing the loss function.

        Returns:
            MeanSquaredErrorLoss: Deserialized loss function instance.

        Raises:
            ValueError: If the loss function name does not match.
        """
        data_dict = json.loads(data)
        if data_dict["name"] != "MeanSquaredErrorLoss":
            raise ValueError(f"Incorrect loss function name: {data_dict['name']}")

        weight_over = data_dict["parameters"].get("weight_over", 1.0)
        weight_under = data_dict["parameters"].get("weight_under", 1.0)

        return cls(weight_over=weight_over, weight_under=weight_under)