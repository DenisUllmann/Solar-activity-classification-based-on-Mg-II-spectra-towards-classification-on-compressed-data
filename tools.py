import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score,confusion_matrix
from irisreader.data.mg2k_centroids import get_mg2k_centroids
from irisreader.data.mg2k_centroids import LAMBDA_MIN as centroid_lambda_min, LAMBDA_MAX as centroid_lambda_max
from scipy.interpolate import interp1d

def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig('Confusion Matrix.png')
    return ax


def to_kcentroid_seq(sample, k=1):
    centroids = get_mg2k_centroids()
    lambda_min = 2793.8401
    lambda_max = 2806.02

    f = interp1d(np.linspace(centroid_lambda_min, centroid_lambda_max, centroids.shape[1]), centroids, kind="cubic")
    centroids_interpolated = f(np.linspace(centroid_lambda_min, centroid_lambda_max, num=200))
    centroids_interpolated /= np.max(centroids_interpolated, axis=1).reshape(-1, 1)

    g = interp1d(np.linspace(lambda_min, lambda_max, sample.shape[1]), sample, kind="cubic")
    sample_interpolated = g(np.linspace(centroid_lambda_min, centroid_lambda_max, num=200))
    sample_interpolated /= np.max(sample_interpolated, axis=1).reshape(-1, 1)

    nc = NearestNeighbors(k).fit(centroids_interpolated, np.arange(len(centroids)))
    k_distances, k_assigned_centroids = nc.kneighbors(sample_interpolated)

    return k_distances, k_assigned_centroids
