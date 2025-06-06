# src/loss_functions/factory.py

from .base import BaseLossFunction
from .binary_cross_entropy import BinaryCrossEntropyLoss
from .mean_squared_error import MeanSquaredErrorLoss
# Import other loss functions as they are implemented

class LossFunctionFactory:
    """
    Factory class to create and manage loss function instances.
    """

    @staticmethod
    def create_loss_function(loss_type: str, **kwargs) -> BaseLossFunction:
        """
        Create a loss function instance based on the loss_type identifier.

        Args:
            loss_type (str): Identifier for the loss function (e.g., 'binary_cross_entropy').
            **kwargs: Additional parameters for the loss function.

        Returns:
            BaseLossFunction: An instance of a loss function.

        Raises:
            ValueError: If the loss type is unsupported.
        """
        loss_type = loss_type.lower()
        if loss_type == "binary_cross_entropy":
            return BinaryCrossEntropyLoss(
                weight_fp=kwargs.get("weight_fp", 1.0),
                weight_fn=kwargs.get("weight_fn", 1.0),
                threshold=kwargs.get("threshold", 0.5)
            )
        elif loss_type == "mean_squared_error":
            return MeanSquaredErrorLoss(
                weight_over=kwargs.get("weight_over", 1.0),
                weight_under=kwargs.get("weight_under", 1.0)
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    @staticmethod
    def deserialize_loss_function(serialized_str: str) -> BaseLossFunction:
        """
        Deserialize a loss function instance from a JSON string.

        Args:
            serialized_str (str): Serialized JSON string representing the loss function.

        Returns:
            BaseLossFunction: Deserialized loss function instance.

        Raises:
            ValueError: If the loss type is unsupported or mismatched.
        """
        import json
        data = json.loads(serialized_str)
        loss_type = data.get("name", "").lower()

        if loss_type == "binarycrossentropyloss":
            return BinaryCrossEntropyLoss.deserialize(serialized_str)
        else:
            raise ValueError(f"Unsupported loss type for deserialization: {loss_type}")
