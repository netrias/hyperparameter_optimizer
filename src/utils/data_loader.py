from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_data():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test
