# src/optimizers/random_search_optimizer.py

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from loss_functions.factory import LossFunctionFactory
from models.base_models import BaseModel

class RandomSearchOptimizer:
    def __init__(self,
                 model_instance,
                 param_distributions,
                 loss_function=None,
                 n_iter=100,
                 scoring='accuracy',
                 random_state=None):
        """
        A simple optimizer using RandomizedSearchCV.

        Args:
            model_instance (BaseModel): Wrapped model instance (must be scikit-learn cloneable).
            param_distributions (dict): Parameter distributions for RandomizedSearchCV.
            loss_function (BaseLossFunction, optional): Custom loss function. Defaults to None.
            n_iter (int, optional): Number of parameter settings sampled. Defaults to 100.
            scoring (str or callable, optional): Scoring strategy. Defaults to 'accuracy'.
            random_state (int, optional): Random state for reproducible searches. Defaults to None.
        """
        self.model = model_instance
        self.param_distributions = param_distributions
        self.loss_function = loss_function
        self.n_iter = n_iter
        self.scoring = scoring
        self.random_state = random_state
        self.cv_results_ = None

    def optimize(self, X_train, y_train):
        """
        Perform hyperparameter optimization using RandomizedSearchCV.
        Returns (best_params, best_score, best_estimator).
        """
        # If a custom loss is provided, wrap it in a scorer
        if self.loss_function:
            def custom_scorer(y_true, y_pred):
                return -self.loss_function.compute(y_true, y_pred)
            scorer = make_scorer(custom_scorer, greater_is_better=True)
        else:
            scorer = self.scoring

        random_search = RandomizedSearchCV(
            estimator=self.model.model,  # the sklearn model inside your BaseModel
            param_distributions=self.param_distributions,
            n_iter=self.n_iter,
            scoring=scorer,
            random_state=self.random_state,
            n_jobs=-1,
            refit=True  # ensure the best_estimator_ is refit
        )

        random_search.fit(X_train, y_train)

        # Save the CV results for inspection if needed
        self.cv_results_ = random_search.cv_results_

        # Update the model with the best estimator found
        self.model.model = random_search.best_estimator_

        # Return all three results
        return (
            random_search.best_params_,
            random_search.best_score_,
            random_search.best_estimator_
        )
