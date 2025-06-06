from sklearn.svm import SVC, SVR
from models.base_models import BaseModel
from sklearn.base import BaseEstimator


class SVCModel(BaseModel, BaseEstimator):
    def __init__(self, model_cls=SVC, loss_function=None, **hyperparameters):
        """
        SVCModel optionally takes model_cls, defaults to sklearn.svm.SVC.
        Hyperparameters can override the defaults for SVC.
        """
        if not hyperparameters:
            hyperparameters = {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}

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

class SVRModel(BaseModel, BaseEstimator):
    def __init__(self, model_cls=SVR, loss_function=None, **hyperparameters):
        """
        SVRModel optionally takes model_cls, defaults to sklearn.svm.SVR.
        Hyperparameters can override the defaults for SVR.
        """
        if not hyperparameters:
            hyperparameters = {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}

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