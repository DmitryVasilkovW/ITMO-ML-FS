from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from fs.params.feature_selection_and_modeling import FeatureSelectionAndModeling


class DimensionalityReducer:
    __x_test_filter = FeatureSelectionAndModeling.get_x_test_filter()
    __pca = PCA(n_components=2)
    __tsne = TSNE(n_components=2, perplexity=min(30, max(5, __x_test_filter.shape[0] // 3)), random_state=42)

    __x_tsne = None
    __x_pca = None

    @classmethod
    def get_x_tsne(cls):
        if cls.__x_tsne is None:
            cls.__x_tsne = cls.__tsne.fit_transform(cls.__x_test_filter)
        return cls.__x_tsne

    @classmethod
    def get_x_pca(cls):
        if cls.__x_pca is None:
            cls.__x_pca = cls.__pca.fit_transform(cls.__x_test_filter)
        return cls.__x_pca
