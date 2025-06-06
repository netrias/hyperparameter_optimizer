# src/loss_functions/binary_cross_entropy.py

import numpy as np
import json 
from .base import BaseLossFunction

class BinaryCrossEntropyLoss(BaseLossFunction):
    """
    Binary Cross-Entropy Loss with customizable penalties and threshold.
    """

    def __init__(self, weight_fp=1.0, weight_fn=1.0, threshold=0.5):
        """
        Initialize the Binary Cross-Entropy loss with weights for false positives,
        false negatives, and a prediction threshold.

        Args:
            weight_fp (float, optional): Weight for false positives. Defaults to 1.0.
            weight_fn (float, optional): Weight for false negatives. Defaults to 1.0.
            threshold (float, optional): Prediction threshold. Defaults to 0.5.
        """
        super().__init__(name="BinaryCrossEntropyLoss")
        self.parameters = {
            "weight_fp": weight_fp,  # Weight for false positives
            "weight_fn": weight_fn,  # Weight for false negatives
            "threshold": threshold   # Prediction threshold
        }

    def compute(self, y_true, y_pred):
        """
        Compute the weighted binary cross-entropy loss.

        Args:
            y_true (array-like): True binary labels.
            y_pred (array-like): Predicted probabilities.

        Returns:
            float: Computed loss value.
        """
        # Ensure y_true and y_pred are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Apply threshold to predictions
        y_pred_thresh = (y_pred >= self.parameters["threshold"]).astype(int)

        # Calculate binary cross-entropy with clipping to avoid log(0)
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = - (
            self.parameters["weight_fn"] * y_true * np.log(y_pred_clipped) +
            self.parameters["weight_fp"] * (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        return np.mean(loss)

    @classmethod
    def deserialize(cls, data: str):
        """
        Deserialize the BinaryCrossEntropyLoss from a JSON string.

        Args:
            data (str): Serialized JSON string representing the loss function.

        Returns:
            BinaryCrossEntropyLoss: Deserialized loss function instance.

        Raises:
            ValueError: If the loss function name does not match.
        """
        data_dict = json.loads(data)
        if data_dict["name"] != "BinaryCrossEntropyLoss":
            raise ValueError(f"Incorrect loss function name: {data_dict['name']}")

        weight_fp = data_dict["parameters"].get("weight_fp", 1.0)
        weight_fn = data_dict["parameters"].get("weight_fn", 1.0)
        threshold = data_dict["parameters"].get("threshold", 0.5)

        return cls(weight_fp=weight_fp, weight_fn=weight_fn, threshold=threshold)
