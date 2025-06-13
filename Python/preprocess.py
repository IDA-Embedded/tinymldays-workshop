import os
from math import ceil

import numpy as np
import pandas as pd
from scipy.io import wavfile

# Sample rate is 16kHz
SAMPLE_RATE = 16000

# FFT frame size is 256 samples (16ms) and stride is 256 samples (16ms)
FRAME_SIZE = 256
FRAME_STRIDE = 256

# Window size is 24 spectral frames (384ms) and stride is 4 frames (64ms)
# We will not run inference at this high rate, but it's a good way to get more training data
WINDOW_SIZE = 24
WINDOW_STRIDE = 4

# Number of frequency bins in the spectrogram
SPECTRUM_SIZE = 67  # We will only use the first 67 bins, which correspond to frequencies up to 4188 Hz

# Mean and std of the spectrogram of all frames for normalization
SPECTRUM_MEAN = 6026.0
SPECTRUM_STD = 14892.0


def preprocess_all(data_dir: str):
    # Load and preprocess all recordings in the data folder
    x_files = []
    y_files = []
    for c_file in os.listdir('../Data/'):
        if c_file.startswith('audio_') and c_file.endswith('.wav'):
            recording_id = c_file[6:-4]
            label_file = 'labels_' + recording_id + '.txt'
            print('Preprocessing ' + c_file + ' and ' + label_file)
            x_file, y_file = _preprocess_recording(data_dir + c_file, data_dir + label_file)
            x_files.append(x_file)
            y_files.append(y_file)

    # Concatenate files into feature and label arrays
    x = np.concatenate(x_files)  # Shape: (number of windows, WINDOW_SIZE, SPECTRUM_SIZE) = (number of windows, 24, 28)
    y = np.concatenate(y_files)  # Shape: (number of windows, 1)

    # Save to files
    os.makedirs('gen/', exist_ok=True)
    np.save('gen/x.npy', x)
    np.save('gen/y.npy', y)


def _preprocess_recording(wav_file: str, label_file: str) -> tuple[np.ndarray, np.ndarray]:
    # Load wav file
    sample_rate, sound_data = wavfile.read(wav_file)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f'Expected sample rate of {SAMPLE_RATE}, but got {sample_rate}')

    # Preprocess data with hamming window and fourier transform
    spectral_frames = []
    for j in range(0, len(sound_data) - FRAME_SIZE, FRAME_STRIDE):
        frame = sound_data[j:j + FRAME_SIZE]
        frame = frame - np.average(frame)
        frame = frame * np.hamming(FRAME_SIZE)
        spectral_frame = np.abs(np.fft.rfft(frame))
        spectral_frame = spectral_frame[:SPECTRUM_SIZE]
        spectral_frames.append(spectral_frame)

    # Convert to numpy array
    spectral_frames = np.array(spectral_frames)

    # Normalize data
    spectral_frames = (spectral_frames - SPECTRUM_MEAN) / SPECTRUM_STD

    # Stack frames into windows
    windows = []
    for i in range(0, len(spectral_frames) - WINDOW_SIZE, WINDOW_STRIDE):
        window = spectral_frames[i:i + WINDOW_SIZE]
        windows.append(window)

    # Convert to numpy array
    x = np.array(windows)

    # Load labels
    labels = pd.read_csv(label_file, sep='\t', header=None, names=['start', 'end', 'label'])

    # Convert time range labels to window labels
    # We want to label windows that are completely within the time range of a label as positive,
    # and windows that are completely outside the time range as negative. Windows that overlap
    # the start and end time are removed, since it's not clear what label they should have, and
    # we don't want to train the model on ambiguous data.
    #
    # Example:
    # Label:              ************************
    # Windows: |   0   | Remove |   1   |   1   | Remove |   0   |
    #              | Remove |   1   |   1   | Remove |   0   |
    y = np.zeros(len(windows))
    start_or_end_overlap = np.zeros(len(windows), dtype=bool)
    window_period = WINDOW_STRIDE * FRAME_STRIDE / sample_rate  # = 4 * 256 / 16000 = 0.064s
    windows_per_second = 1 / window_period  # = 15.625
    window_length = WINDOW_SIZE * FRAME_STRIDE / sample_rate  # = 24 * 256 / 16000 = 0.384s
    for index, label in labels.iterrows():
        # Compute index of start/end of label in windows
        start_window = ceil(label['start'] * windows_per_second)
        if start_window >= len(y):
            continue
        end_window = ceil((label['end'] - window_length) * windows_per_second)

        # Label all windows within the label range as positive
        y[start_window:end_window] = 1

        # Mark for removal all windows that overlap the start of a range label
        i = start_window - 1
        while i >= 0 and i * window_period + window_length > label['start']:
            start_or_end_overlap[i] = True
            i -= 1

        # Mark for removal all windows that overlap the end of a range label
        i = end_window
        while i < len(y) and i * window_period < label['end']:
            start_or_end_overlap[i] = True
            i += 1

    # Remove windows that overlap the start and end of a range label
    x = x[~start_or_end_overlap]
    y = y[~start_or_end_overlap]

    return x, y


if __name__ == '__main__':
    preprocess_all('../Data/')
