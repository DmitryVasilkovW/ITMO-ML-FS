from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

from fs.dataset.axis_repo import DataRepo
from fs.params.feature_selection_and_modeling import FeatureSelectionAndModeling


class ClusteringEvaluator:
    __kmeans = KMeans(n_clusters=2, random_state=42)

    __x_test = DataRepo.get_axis('x', 'test')
    __x_test_filter = FeatureSelectionAndModeling.get_x_test_filter()
    __y_test = DataRepo.get_axis('y', 'test')

    __clusters_before = __kmeans.fit_predict(__x_test)
    __clusters_after = __kmeans.fit_predict(__x_test_filter)

    @classmethod
    def get_silhouette_before(cls):
        return silhouette_score(cls.__x_test, cls.__clusters_before)

    @classmethod
    def get_silhouette_after(cls):
        return silhouette_score(cls.__x_test_filter, cls.__clusters_after)

    @classmethod
    def get_adjusted_rand_before(cls):
        return adjusted_rand_score(cls.__y_test, cls.__clusters_before)

    @classmethod
    def get_adjusted_rand_after(cls):
        return adjusted_rand_score(cls.__y_test, cls.__clusters_after)
