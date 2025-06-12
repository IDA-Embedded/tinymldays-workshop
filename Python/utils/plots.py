import keras
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from preprocess import *


def plot_dataset(x: np.ndarray, y: np.ndarray, block: bool):
    # Concatenate the windows along the time axis
    # However, there are overlapping windows, so we should only use the first WINDOW_STRIDE frames of each window and skip the rest.
    x_concatenated = np.concatenate(x[:, 0:WINDOW_STRIDE, :], axis=0)

    # Limit values to [-2, 2] for better visualization
    x_concatenated = np.clip(x_concatenated, -2, 2)

    # Create plot and clear axes
    plt.figure(figsize=(12, 6))
    plt.title('Spectrogram of entire data set')
    plt.xticks([])
    plt.yticks([])

    # Plot the spectrogram
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax1.imshow(x_concatenated.T, aspect='auto', origin='lower', interpolation='none')
    ax1.set_ylabel('Frequency bin')
    ax1.set_xticks([])
    tick_frames = np.arange(0, len(x_concatenated), 60 * 5 * SAMPLE_RATE / FRAME_STRIDE)
    tick_labels = np.round(tick_frames * FRAME_STRIDE / SAMPLE_RATE / 60, 2)
    ax1.set_xticks(tick_frames, labels=tick_labels)
    ax1.set_xticklabels(tick_labels)

    # Plot the labels
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax2.plot(y)
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Label')
    tick_frames = np.arange(0, len(y), 60 * 5 * SAMPLE_RATE / FRAME_STRIDE / WINDOW_STRIDE)
    tick_labels = np.round(tick_frames * WINDOW_STRIDE * FRAME_STRIDE / SAMPLE_RATE / 60, 2)
    ax2.set_xticks(tick_frames)
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlim(0, len(y) - 1)
    ax2.set_yticks([])

    # Show plot
    plt.tight_layout()
    plt.show(block=block)


def plot_tsne(x: np.ndarray, y: np.ndarray, block: bool):
    # Flatten the windows to 2D
    x_flat = x.reshape(x.shape[0], -1)

    # Take a random sample of 5000 points to speed up computation
    if x_flat.shape[0] > 5000:
        indices = np.random.choice(x_flat.shape[0], 5000, replace=False)
        x_flat = x_flat[indices]
        y = y[indices]

    # Use PCA to reduce dimensionality to max 128 dimensions to speed up computation
    if x_flat.shape[1] > 128:
        pca = PCA(n_components=128)
        x_flat = pca.fit_transform(x_flat)

    # Compute t-SNE
    print('Computing t-SNE...')
    tsne = TSNE(n_components=2)
    x_tsne = tsne.fit_transform(x_flat)

    # Create scatter plot
    # y is either 0 or 1 - use blue for 0 and orange for 1
    plt.figure(figsize=(12, 8))
    plt.scatter(x_tsne[y == 0, 0], x_tsne[y == 0, 1], color='blue', label='Negative', alpha=0.5)
    plt.scatter(x_tsne[y == 1, 0], x_tsne[y == 1, 1], color='orange', label='Positive', alpha=0.5)
    plt.title('t-SNE projection of entire dataset')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.show(block=block)


def plot_learning_curves(model: keras.models.Model, block: bool):
    plt.figure(figsize=(12, 6))
    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title('Learning curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training loss', 'Validation loss'])
    plt.show(block=block)


def plot_predictions_vs_labels(y_pred: np.ndarray, y_test: np.ndarray, block: bool):
    plt.figure(figsize=(12, 6))
    plt.plot(y_pred, color='blue')
    plt.plot(-y_test, color='green')  # Negate labels to make them easier to see
    plt.title('Test set predictions vs labels')
    plt.xlabel('Window')
    plt.ylabel('Prediction')
    plt.legend(['Prediction', 'Label'])
    plt.show(block=block)
