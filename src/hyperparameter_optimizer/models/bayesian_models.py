# src/optimizers/bayesian_models.py

from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import BayesianRidge
from models.base_models import BaseModel
from sklearn.base import BaseEstimator


class GaussianNBModel(BaseModel, BaseEstimator):
    def __init__(self, model_cls=GaussianNB, loss_function=None, **hyperparameters):
        """
        GaussianNBModel optionally takes model_cls, defaults to GaussianNB.
        Hyperparameters can override defaults like var_smoothing.
        """
        if not hyperparameters:
            hyperparameters = {'var_smoothing': 1e-9}

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


class BernoulliNBModel(BaseModel, BaseEstimator):
    def __init__(self, model_cls=BernoulliNB, loss_function=None, **hyperparameters):
        """
        BernoulliNBModel optionally takes model_cls, defaults to BernoulliNB.
        Hyperparameters can override defaults like alpha, binarize, etc.
        """
        if not hyperparameters:
            hyperparameters = {'alpha': 1.0, 'binarize': 0.0, 'fit_prior': True}

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


class BayesianRidgeModel(BaseModel, BaseEstimator):
    def __init__(self, model_cls=BayesianRidge, loss_function=None, **hyperparameters):
        """
        BayesianRidgeModel optionally takes model_cls, defaults to BayesianRidge.
        Hyperparameters can override defaults like n_iter, alpha_1, alpha_2, etc.
        """
        if not hyperparameters:
            hyperparameters = {
                'max_iter': 300,
                'tol': 0.001,
                'alpha_1': 1e-6,
                'alpha_2': 1e-6,
                'lambda_1': 1e-6,
                'lambda_2': 1e-6
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