from matplotlib import pyplot as plt

from fs.dataset.axis_repo import DataRepo
from fs.feature_reduction.dimensionality_reducer import DimensionalityReducer

x_pca = DimensionalityReducer.get_x_pca()
x_tsne = DimensionalityReducer.get_x_tsne()
y_test = DataRepo.get_axis('y', 'test')


def show_plot():
    plt.figure(figsize=(12, 6))

    unique_classes = set(y_test)
    classes = ['spam', 'ham']
    colors = ['blue', 'green']
    color_map = {cls: colors[i % len(colors)] for i, cls in enumerate(unique_classes)}

    plt.subplot(1, 2, 1)
    for cls in unique_classes:
        indices = [i for i, label in enumerate(y_test) if label == cls]
        plt.scatter(
            x_pca[indices, 0],
            x_pca[indices, 1],
            label=f"{classes[cls]}",
            color=color_map[cls],
            alpha=0.7,
            edgecolor='k'
        )
    plt.title("PCA Visualization")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Classes")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for cls in unique_classes:
        indices = [i for i, label in enumerate(y_test) if label == cls]
        plt.scatter(
            x_tsne[indices, 0],
            x_tsne[indices, 1],
            label=f"{classes[cls]}",
            color=color_map[cls],
            alpha=0.7,
            edgecolor='k'
        )
    plt.title("t-SNE Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(title="Classes")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
