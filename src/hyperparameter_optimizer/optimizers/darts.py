import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import pandas as pd

class DARTSOptimizer:
    def __init__(self, input_size, max_layers=10, max_nodes=None):
        self.input_size = input_size
        self.max_layers = max_layers
        self.max_nodes = max_nodes if max_nodes else input_size
        self.trials = Trials()

    def build_model(self, architecture):
        layers = []
        input_dim = self.input_size
        for i in range(architecture['num_layers']):
            output_dim = min(input_dim, architecture['num_nodes'][i])
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim
        layers.append(nn.Linear(input_dim, 1))  # Single output for binary classification
        model = nn.Sequential(*layers)
        return model

    def objective(self, architecture):
        model = self.build_model(architecture)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()

        dataset = TensorDataset(self.X_train, self.y_train)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(10):
            for data, target in loader:
                target = target.unsqueeze(1)  # Ensure correct shape
                optimizer.zero_grad()
                output = model(data.float())
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Evaluate the model for accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in DataLoader(TensorDataset(self.X_test, self.y_test), batch_size=64):
                target = target.unsqueeze(1)  # Ensure correct shape
                outputs = model(data.float())
                predicted = (outputs > 0).float()  # Convert logits to class labels
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = correct / total

        return {'loss': -accuracy, 'status': STATUS_OK, 'accuracy': accuracy}

    def optimize(self, X_train, y_train, X_test, y_test):
        # Explicit conversion from DataFrame/Series to numpy arrays
        X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train = y_train.values if isinstance(y_train, pd.DataFrame) else y_train
        X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        y_test = y_test.values if isinstance(y_test, pd.DataFrame) else y_test

        # Convert numpy arrays to torch tensors
        self.X_train = torch.tensor(X_train, dtype=torch.float)
        self.y_train = torch.tensor(y_train, dtype=torch.float).squeeze()
        self.X_test = torch.tensor(X_test, dtype=torch.float)
        self.y_test = torch.tensor(y_test, dtype=torch.float).squeeze()

        # Define the search space
        search_space = {
            'num_layers': hp.choice('num_layers', range(1, self.max_layers + 1)),
            'num_nodes': hp.choice('num_nodes', [list(range(1, self.max_nodes + 1)) for _ in range(self.max_layers)])
        }
        
        best = fmin(fn=self.objective, space=search_space, algo=tpe.suggest, max_evals=50, trials=self.trials)
        best_score = -min(t['result']['loss'] for t in self.trials.trials)  # Since we minimized the negative accuracy
        best_accuracy = max(t['result']['accuracy'] for t in self.trials.trials)  # Retrieve the maximum accuracy
        
        best_architecture = {key: search_space[key][value] for key, value in best.items()}
        return best_architecture, best_score, best_accuracy


