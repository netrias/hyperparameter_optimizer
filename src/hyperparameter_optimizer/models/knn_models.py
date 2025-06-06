# src/models/knn_models.py

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from models.base_models import BaseModel
from sklearn.base import BaseEstimator


class KNNClassifierModel(BaseModel, BaseEstimator):
    def __init__(self, model_cls=KNeighborsClassifier, loss_function=None, **hyperparameters):
        """
        Initialize the KNNClassifierModel with a model_cls (defaults to KNeighborsClassifier),
        hyperparameters, and an optional loss function.

        Args:
            model_cls (class, optional): Class to be used as the underlying KNN classifier.
            hyperparameters (dict, optional): Hyperparameters for the classifier.
            loss_function (BaseLossFunction, optional): Custom loss function instance.
        """
        if not hyperparameters:
            hyperparameters = {
                'n_neighbors': 5,
                'weights': 'uniform',
                'algorithm': 'auto'
            }

        self.model_cls = model_cls
        self.hyperparameters = hyperparameters
        self.loss_function = loss_function
        self.model = self.model_cls(**self.hyperparameters)

        super().__init__(model_cls, hyperparameters, loss_function=loss_function)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def get_params(self, deep=True):
        """
        Return model parameters in a dictionary format.
        Ensures compatibility with GridSearchCV.
        """
        return self.hyperparameters

    def set_params(self, **params):
        """
        Update model parameters and ensure they are properly set in the model instance.
        """
        self.hyperparameters.update(params)
        self.model = self.model_cls(**self.hyperparameters)  # Reinitialize model with new params
        return self


class KNNRegressorModel(BaseModel, BaseEstimator):
    def __init__(self, model_cls=KNeighborsRegressor, loss_function=None, **hyperparameters):
        """
        Initialize the KNNRegressorModel with a model_cls (defaults to KNeighborsRegressor),
        hyperparameters, and an optional loss function.

        Args:
            model_cls (class, optional): Class to be used as the underlying KNN regressor.
            hyperparameters (dict, optional): Hyperparameters for the regressor.
            loss_function (BaseLossFunction, optional): Custom loss function instance.
        """
        if not hyperparameters:
            hyperparameters = {
                'n_neighbors': 5,
                'weights': 'uniform',
                'algorithm': 'auto'
            }

        self.model_cls = model_cls
        self.hyperparameters = hyperparameters
        self.loss_function = loss_function
        self.model = self.model_cls(**self.hyperparameters)

        super().__init__(model_cls, hyperparameters, loss_function=loss_function)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def get_params(self, deep=True):
        """
        Return model parameters in a dictionary format.
        Ensures compatibility with GridSearchCV.
        """
        return self.hyperparameters

    def set_params(self, **params):
        """
        Update model parameters and ensure they are properly set in the model instance.
        """
        self.hyperparameters.update(params)
        self.model = self.model_cls(**self.hyperparameters)  # Reinitialize model with new params
        return self