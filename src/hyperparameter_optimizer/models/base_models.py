# src/models/base_models.py

from sklearn.base import BaseEstimator

class BaseModel(BaseEstimator):
    def __init__(self, model_cls, hyperparameters=None, loss_function=None):
        if hyperparameters is None:
            hyperparameters = {}
        self.model_cls = model_cls
        self.hyperparameters = hyperparameters
        self.loss_function = loss_function

        # Instantiate the actual sklearn model
        self.model = self.model_cls(**self.hyperparameters)

    def fit(self, X_train, y_train):
        """
        Bandaid fix:
        If y_train is None (unsupervised scenario), just call .fit(X_train).
        Otherwise, call .fit(X_train, y_train) for supervised models.
        """
        if y_train is None:
            self.model.fit(X_train)
        else:
            self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        """
        Use a custom loss if provided, else use the model's default .score().
        For unsupervised, this might ignore y or do something custom.
        """
        if self.loss_function is not None and y is not None:
            y_pred = self.predict(X)
            return self.loss_function.compute(y, y_pred)
        else:
            return self.model.score(X, y)

    def get_params(self, deep=True):
        return {
            'model_cls': self.model_cls,
            'hyperparameters': self.hyperparameters,
            'loss_function': self.loss_function
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        if 'hyperparameters' in params:
            self.model = self.model_cls(**self.hyperparameters)
        return self
