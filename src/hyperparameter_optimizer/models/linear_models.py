# src/models/logistic_models.py

from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from .base_models import BaseModel
from sklearn.base import BaseEstimator


class LogisticModel(BaseModel, BaseEstimator):
    def __init__(self, model_cls=LogisticRegression, loss_function=None, **hyperparameters):
        """
        LogisticModel optionally takes model_cls, defaults to sklearn.linear_model.LogisticRegression.
        Hyperparameters can override defaults like max_iter, penalty, C, etc.
        """
        if not hyperparameters:
            hyperparameters = {'max_iter': 200, 'penalty': 'l2', 'C': 1.0}

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


class RidgeModel(BaseModel, BaseEstimator):
    def __init__(self, model_cls=Ridge, loss_function=None, **hyperparameters):
        """
        RidgeModel optionally takes model_cls, defaults to sklearn.linear_model.Ridge.
        Hyperparameters can override defaults like alpha.
        """
        if not hyperparameters:
            hyperparameters = {'alpha': 1.0}

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


class LassoModel(BaseModel, BaseEstimator):
    def __init__(self, model_cls=Lasso, loss_function=None, **hyperparameters):
        """
        LassoModel optionally takes model_cls, defaults to sklearn.linear_model.Lasso.
        Hyperparameters can override defaults like alpha.
        """
        if not hyperparameters:
            hyperparameters = {'alpha': 1.0}

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