# src/optimizers/bayesian_models.py

#from skopt import BayesSearchCV
from sklearn.metrics import make_scorer
from loss_functions.factory import LossFunctionFactory
from models.base_models import BaseModel

class BayesSearchOptimizer:
    def __init__(self, model_instance, param_grid, loss_function=None, scoring='accuracy', n_iter=50, cv=5, random_state=42):
        self.model = model_instance
        self.param_grid = param_grid
        self.loss_function = loss_function
        self.scoring = scoring
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self.cv_results_ = None

    def optimize(self, X_train, y_train):
        if self.loss_function:
            def custom_scorer(y_true, y_pred):
                return -self.loss_function.compute(y_true, y_pred)
            scorer = make_scorer(custom_scorer, greater_is_better=True)
        else:
            scorer = self.scoring

        bayes_search = BayesSearchCV(
            estimator=self.model.model,
            search_spaces=self.param_grid,
            n_iter=self.n_iter,
            scoring=scorer,
            cv=self.cv,
            random_state=self.random_state,
            n_jobs=-1
        )
        bayes_search.fit(X_train, y_train)
        self.model.model = bayes_search.best_estimator_
        self.cv_results_ = bayes_search.cv_results_
        best_score = bayes_search.best_score_
        best_params = bayes_search.best_params_
        return best_params, best_score
