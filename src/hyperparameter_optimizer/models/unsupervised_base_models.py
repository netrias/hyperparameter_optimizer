# src/models/unsupervised_base_model.py

from sklearn.base import BaseEstimator

class UnsupervisedBaseModel(BaseEstimator):
    """
    A base class for unsupervised estimators (clustering, etc.).
    Typically does not use labels. 
    """

    def __init__(self, model_cls, hyperparameters=None, loss_function=None):
        if hyperparameters is None:
            hyperparameters = {}
        self.model_cls = model_cls
        self.hyperparameters = hyperparameters
        self.loss_function = loss_function
        self.model = self.model_cls(**self.hyperparameters)

    def fit(self, X, y=None):
        """
        Unsupervised fit typically ignores y.
        """
        self.model.fit(X)
        return self

    def predict(self, X):
        """
        Some clustering models (KMeans) define .predict, while others (DBSCAN) might not.
        Adjust or handle exceptions as needed.
        """
        return self.model.predict(X)

    def score(self, X, y=None):
        """
        If a custom loss_function is defined, handle it. 
        If not, rely on the model's .score(X) if it exists (e.g. KMeans).
        """
        if self.loss_function is not None and y is not None:
            # If you define a custom metric needing X,y, do it here.
            y_pred = self.predict(X)
            return self.loss_function.compute(y, y_pred)
        else:
            # e.g., KMeans uses negative inertia, ignoring y
            return self.model.score(X)

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
