from fs.evaluator.clustering_evaluator import ClusteringEvaluator


def show_data():
    print(f'Adjusted Rand Index before selecting features: {ClusteringEvaluator.get_adjusted_rand_before()}')
    print(f'Adjusted Rand Index after selecting traits: {ClusteringEvaluator.get_adjusted_rand_after()}')
    print(f'Silhouette before selecting features: {ClusteringEvaluator.get_silhouette_before()}')
    print(f'Silhouette after selecting the traits: {ClusteringEvaluator.get_silhouette_after()}')
