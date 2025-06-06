# src/models/ensemble_models.py

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from .base_models import BaseModel
from sklearn.base import BaseEstimator


class RandomForestModel(BaseModel, BaseEstimator):
    def __init__(self, model_cls=RandomForestClassifier, loss_function=None, **hyperparameters):
        """
        By default, model_cls is RandomForestClassifier, but this allows 'model_cls'
        to be recognized as a parameter. Hyperparameters can override defaults like
        n_estimators, max_depth, etc.
        """
        if not hyperparameters:
            hyperparameters = {'n_estimators': 100, 'max_depth': None}

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


class GradientBoostingModel(BaseModel, BaseEstimator):
    def __init__(self, model_cls=GradientBoostingClassifier, loss_function=None, **hyperparameters):
        """
        GradientBoostingModel optionally takes model_cls, defaults to GradientBoostingClassifier.
        Hyperparameters can override defaults like n_estimators, learning_rate, max_depth, etc.
        """
        if not hyperparameters:
            hyperparameters = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3
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