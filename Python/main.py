import os
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout

from preprocess import preprocess_all, WINDOW_SIZE, SPECTRUM_SIZE
from utils.export_tflite import write_model_c_file
from utils.plots import plot_dataset, plot_tsne, plot_predictions_vs_labels, plot_learning_curves

# Minimize TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


def preprocess_and_load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Preprocess data if not done already
    if not os.path.exists('gen/x.npy') or not os.path.exists('gen/y.npy'):
        preprocess_all('../Data/')

    # Load preprocessed data
    x = np.load('gen/x.npy')
    y = np.load('gen/y.npy')

    # Plot the dataset as spectrogram of the entire dataset with labels underneath and as t-SNE projection
    plot_dataset(x, y, block=False)
    plot_tsne(x, y, block=False)

    # Shuffle the data
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    # Split into training, validation and test sets
    x_train, x_val, x_test = np.split(x, [int(.6 * len(x)), int(.8 * len(x))])
    y_train, y_val, y_test = np.split(y, [int(.6 * len(y)), int(.8 * len(y))])

    return x_train, y_train, x_val, y_val, x_test, y_test


def train_model(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray):
    # Determine negative to positive ratio
    num_positives = np.sum(y_train)
    num_negatives = len(y_train) - num_positives
    ratio = num_negatives / num_positives
    print('Negative to positive ratio: ', ratio)

    # Build and compile model
    print('Building model...')
    model = Sequential()
    model.add(Conv1D(16, 5, activation='relu', input_shape=(WINDOW_SIZE, SPECTRUM_SIZE)))  # Output shape (22, 8)
    model.add(MaxPooling1D(2))  # Output shape (11, 8)
    model.add(Dropout(0.1))
    model.add(Conv1D(16, 5, activation='relu'))  # Output shape (9, 8)
    model.add(MaxPooling1D(2))  # Output shape (4, 8)
    model.add(Dropout(0.1))
    model.add(Flatten())  # Output shape (32)
    model.add(Dense(1, activation='sigmoid'))  # Output shape (1)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Print model summary
    model.summary()

    # Train model with early stopping; save best model
    print('Training model...')
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=16)
    model_checkpoint = keras.callbacks.ModelCheckpoint('gen/model.keras', monitor='val_loss', save_best_only=True)
    model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_val, y_val),
              callbacks=[early_stopping, model_checkpoint])

    # Plot learning curves
    plot_learning_curves(model, block=False)

    # Load and return best model
    model = keras.models.load_model('gen/model.keras')
    return model


def evaluate_model(model: keras.models.Model, x_val: np.ndarray, y_val: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
    # Evaluate model on validation and test sets
    val_loss, val_accuracy = model.evaluate(x_val, y_val)
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)

    # Print evaluation metrics
    print()
    print('Validation loss:     %.4f' % val_loss)
    print('Validation accuracy: %.4f' % val_accuracy)
    print('Test loss:           %.4f' % test_loss)
    print('Test accuracy:       %.4f' % test_accuracy)

    # Print confusion matrix
    y_pred_bool = np.round(y_pred)
    confusion_matrix = np.zeros((2, 2))
    for i in range(len(y_pred_bool)):
        confusion_matrix[int(y_test[i]), int(y_pred_bool[i, 0])] += 1
    print('True positives:     ', int(confusion_matrix[1, 1]))
    print('True negatives:     ', int(confusion_matrix[0, 0]))
    print('False positives:    ', int(confusion_matrix[0, 1]))
    print('False negatives:    ', int(confusion_matrix[1, 0]))

    # Plot predictions vs labels
    plot_predictions_vs_labels(y_pred, y_test, block=True)


def export_model_to_tflite(model: keras.models.Model, x_train: np.ndarray, enable_quantization: bool = True):
    # Convert to TensorFlow Lite model
    print('Converting to TensorFlow Lite model...')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if enable_quantization:
        # Function for generating representative data
        def representative_dataset():
            x_train_samples = x_train[np.random.choice(x_train.shape[0], 5000, replace=False)]
            yield [x_train_samples.astype(np.float32)]

        # Quantize model
        print("Quantizing TensorFlow Lite model...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    # Print quantization scale and zero point
    if enable_quantization:
        # Load model in interpreter
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Do print
        print("Input scale:", input_details[0]['quantization'][0])
        print("Input zero point:", input_details[0]['quantization'][1])
        print("Output scale:", output_details[0]['quantization'][0])
        print("Output zero point:", output_details[0]['quantization'][1])

    # Export TensorFlow Lite model to C source files
    print("Exporting TensorFlow Lite model to C source file...")
    write_model_c_file("../ESP-32/main/model.c", tflite_model)

    # Save TensorFlow Lite model
    with open(f"gen/model.tflite", "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    # Preprocess and load data
    x_train, y_train, x_val, y_val, x_test, y_test = preprocess_and_load_data()

    # Train model
    model = train_model(x_train, y_train, x_val, y_val)

    # Evaluate model
    evaluate_model(model, x_val, y_val, x_test, y_test)

    # Save TFLite model
    export_model_to_tflite(model, x_train)

    print("Done.")
