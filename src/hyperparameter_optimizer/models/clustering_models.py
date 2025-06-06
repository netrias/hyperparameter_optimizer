# src/optimizers/clustering_models.py

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
from .unsupervised_base_models import UnsupervisedBaseModel

class KMeansModel(UnsupervisedBaseModel):
    def __init__(self, model_cls=KMeans, hyperparameters=None, loss_function=None):
        """
        KMeansModel optionally takes model_cls, defaults to sklearn.cluster.KMeans.
        hyperparameters can override defaults like n_clusters, init, n_init, max_iter, etc.
        """
        if hyperparameters is None:
            hyperparameters = {
                'n_clusters': 8,
                'init': 'k-means++',
                'n_init': 10,
                'max_iter': 300,
                'random_state': 42
            }
        super().__init__(model_cls, hyperparameters, loss_function=loss_function)

class DBSCANModel(UnsupervisedBaseModel):
    def __init__(self, model_cls=DBSCAN, hyperparameters=None, loss_function=None):
        if hyperparameters is None:
            hyperparameters = {'eps': 0.5, 'min_samples': 5}
        super().__init__(model_cls, hyperparameters, loss_function=loss_function)

    def score(self, X, y=None):
        """
        Provide a dummy or custom clustering score method 
        so scikit-learn won't complain that DBSCAN lacks .score().
        
        Options:
          1) Return a dummy value (0.0) 
          2) Use a real cluster metric like silhouette_score (requires fit_predict).
        """
        # ---- Option A: Dummy approach (always 0) ----
        # return 0.0
        
        # ---- Option B: Real silhouette approach (only valid if > 1 cluster) ----
        labels = self.model.fit_predict(X)
        # If DBSCAN lumps everything into one cluster or all noise, silhouette is undefined.
        if len(set(labels)) < 2:
            return -1  # or 0.0, some fallback
        return silhouette_score(X, labels)




# src/models/clustering_models.py

class AgglomerativeClusteringModel(UnsupervisedBaseModel):
    def __init__(self, model_cls=AgglomerativeClustering, hyperparameters=None, loss_function=None):
        if hyperparameters is None:
            hyperparameters = {
                'n_clusters': 8,
                'metric': 'euclidean',  # Updated from 'affinity' to 'metric'
                'linkage': 'ward'
            }
        super().__init__(model_cls, hyperparameters, loss_function=loss_function)

    def score(self, X, y=None):
        """
        Provide a dummy score method to satisfy GridSearchCV.
        Returns silhouette score as an example.
        """
        from sklearn.metrics import silhouette_score

        # Predict cluster labels
        labels = self.predict(X)

        # Check if there are at least 2 clusters
        if len(set(labels)) < 2 or len(set(labels)) == len(X):
            return -1  # Undefined silhouette score

        return silhouette_score(X, labels)

class SpectralClusteringModel(UnsupervisedBaseModel):
    def __init__(self, model_cls=SpectralClustering, hyperparameters=None, loss_function=None):
        """
        SpectralClusteringModel optionally takes model_cls, defaults to sklearn.cluster.SpectralClustering.
        hyperparameters can override defaults like n_clusters, eigen_solver, n_init, gamma, etc.
        """
        if hyperparameters is None:
            hyperparameters = {
                'n_clusters': 8,
                'eigen_solver': None,
                'n_init': 10,
                'gamma': 1.0
            }
        super().__init__(model_cls, hyperparameters, loss_function=loss_function)
