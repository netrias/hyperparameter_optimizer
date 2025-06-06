# src/optimizers/grid_search.py

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from loss_functions.factory import LossFunctionFactory  # Removed 'src.' prefix
from models.base_models import BaseModel  # Removed 'src.' prefix

class GridSearchOptimizer:
    def __init__(self, model_instance, param_grid, loss_function=None, cv=5, scoring='accuracy'):
        """
        Initialize the GridSearchOptimizer with a model instance, parameter grid,
        optional custom loss function, cross-validation folds, and scoring metric.

        Args:
            model_instance (BaseModel): The model to optimize.
            param_grid (dict): Hyperparameter grid for GridSearchCV.
            loss_function (BaseLossFunction, optional): Custom loss function instance.
            cv (int, optional): Number of cross-validation folds. Defaults to 5.
            scoring (str or callable, optional): Scoring metric. Defaults to 'accuracy'.
        """
        self.model = model_instance
        self.param_grid = param_grid
        self.loss_function = loss_function
        self.cv = cv
        self.scoring = scoring
        self.cv_results_ = None

    def optimize(self, X_train, y_train):
        print("Before GridSearchCV:", self.model.get_params())
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        print("After GridSearchCV:", grid_search.best_estimator_.get_params())

        self.cv_results_ = grid_search.cv_results_
        return grid_search.best_params_, grid_search.best_score_, grid_search.best_estimator_
