from models.linear_models import LogisticModel
from models.ensemble_models import RandomForestModel
from optimizers.grid_search import GridSearchOptimizer
from utils.data_loader import load_data

def run_demo():
    X_train, X_test, y_train, y_test = load_data()

    # Logistic Regression Model
    logistic_params = {'C': [0.1, 1, 10], 'max_iter': [100, 200]}
    logistic_model = LogisticModel(hyperparameters={})
    optimizer = GridSearchOptimizer(logistic_model, logistic_params)
    best_params_logistic = optimizer.optimize(X_train, y_train)
    print("Best params for Logistic Regression:", best_params_logistic)
    print("Test score:", logistic_model.score(X_test, y_test))

    # Random Forest Model
    rf_params = {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 10]}
    rf_model = RandomForestModel(hyperparameters={})
    optimizer = GridSearchOptimizer(rf_model, rf_params)
    best_params_rf = optimizer.optimize(X_train, y_train)
    print("Best params for Random Forest:", best_params_rf)
    print("Test score:", rf_model.score(X_test, y_test))

if __name__ == '__main__':
    run_demo()
