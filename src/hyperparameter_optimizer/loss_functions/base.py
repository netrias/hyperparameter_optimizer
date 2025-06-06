# src/loss_functions/base.py

from abc import ABC, abstractmethod
import json

class BaseLossFunction(ABC):
    """
    Abstract base class for custom loss functions.
    """

    def __init__(self, name: str):
        """
        Initialize the loss function with a name and a parameters dictionary.
        """
        self.name = name
        self.parameters = {}

    @abstractmethod
    def compute(self, y_true, y_pred):
        """
        Compute the loss given true and predicted values.
        Must be implemented by subclasses.
        """
        pass

    def update_parameters(self, **kwargs):
        """
        Update the parameters of the loss function.
        """
        for key, value in kwargs.items():
            if key in self.parameters:
                self.parameters[key] = value
            else:
                raise ValueError(f"Parameter '{key}' not found in loss function '{self.name}'.")

    def serialize(self) -> str:
        """
        Serialize the loss function to a JSON string.
        """
        data = {
            "name": self.name,
            "parameters": self.parameters
        }
        return json.dumps(data)

    @classmethod
    @abstractmethod
    def deserialize(cls, data: str):
        """
        Deserialize the loss function from a JSON string.
        Must be implemented by subclasses.
        """
        pass
