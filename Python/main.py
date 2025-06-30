import os
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout

from generate_test_case import generate_test_case
from preprocess import preprocess_all, WINDOW_SIZE, SPECTRUM_SIZE, SPECTRUM_TOP, SPECTRUM_SRC, SPECTRUM_DST, SPECTRUM_MEAN, SPECTRUM_STD, SAMPLE_RATE, FRAME_SIZE, FRAME_STRIDE
from utils.export_tflite import write_model_h_file, write_model_c_file
from utils.plots import plot_dataset, plot_tsne, plot_predictions_vs_labels, plot_learning_curves

# Minimize TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

USE_CACHED_DATA = False  # Set to True to reuse cached data, False to force preprocess data


def preprocess_and_load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Preprocess data if not done already
    if not USE_CACHED_DATA \
            or not os.path.exists('gen/x_train.npy') or not os.path.exists('gen/y_train.npy') \
            or not os.path.exists('gen/x_val.npy') or not os.path.exists('gen/y_val.npy') \
            or not os.path.exists('gen/x_test.npy') or not os.path.exists('gen/y_test.npy'):
        preprocess_all('../Data/')

    # Load preprocessed data
    x_train = np.load('gen/x_train.npy')
    y_train = np.load('gen/y_train.npy')
    x_val = np.load('gen/x_val.npy')
    y_val = np.load('gen/y_val.npy')
    x_test = np.load('gen/x_test.npy')
    y_test = np.load('gen/y_test.npy')

    # Concatenate splits into one data set for statistics and plotting
    x = np.concatenate([x_train, x_val, x_test])  # Shape: (number of windows, WINDOW_SIZE, SPECTRUM_SIZE) = (number of windows, 24, 28)
    y = np.concatenate([y_train, y_val, y_test])  # Shape: (number of windows, 1)

    # Print mean and std of the dataset
    print('Mean of the dataset:', np.mean(x))
    print('Standard deviation of the dataset:', np.std(x))

    # Plot the dataset as spectrogram of the entire dataset with labels underneath and as t-SNE projection
    # plot_dataset(x, y, block=True)
    # plot_tsne(x, y, block=True)

    # Shuffle the training data
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

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


def export_model_to_tflite(model: keras.models.Model, x_train: np.ndarray, enable_quantization: bool = True) -> object:
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
    print("Exporting TensorFlow Lite model to C source files...")
    defines = {
        "SAMPLE_RATE": SAMPLE_RATE,
        "FRAME_SIZE": FRAME_SIZE,
        "FRAME_STRIDE": FRAME_STRIDE,
        "WINDOW_SIZE": WINDOW_SIZE,
        "SPECTRUM_TOP": SPECTRUM_TOP,
        "SPECTRUM_SIZE": SPECTRUM_SIZE,
        "SPECTRUM_MEAN": SPECTRUM_MEAN,
        "SPECTRUM_STD": SPECTRUM_STD
    }
    declarations = [
        "const unsigned long SPECTRUM_SRC[] = { " + ", ".join(map(str, SPECTRUM_SRC)) + " };",
        "const unsigned long SPECTRUM_DST[] = { " + ", ".join(map(str, SPECTRUM_DST)) + " };"
    ]
    write_model_h_file("../ESP32/main/model.h", defines, declarations)
    write_model_c_file("../ESP32/main/model.c", tflite_model)

    # Save TensorFlow Lite model
    with open(f"gen/model.tflite", "wb") as f:
        f.write(tflite_model)

    return tflite_model


def evaluate_tflite_model(tflite_model: object, x_test: np.ndarray, y_test: np.ndarray):
    # Load interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zero_point = input_details[0]["quantization"]
    output_scale, output_zero_point = output_details[0]["quantization"]

    # Quantize x_test
    x_test_quantized = x_test / input_scale + input_zero_point
    x_test_quantized = np.clip(x_test_quantized, -128, 127)
    x_test_quantized_int = x_test_quantized.astype(np.int8)

    # Predict
    y_pred_quantized = np.empty((len(x_test_quantized_int), 1, 1), dtype=np.int8)
    for i in range(len(x_test_quantized_int)):
        interpreter.set_tensor(
            input_details[0]["index"], x_test_quantized_int[i].reshape(1, 24, 28)
        )
        interpreter.invoke()
        y_pred_quantized[i] = interpreter.get_tensor(output_details[0]["index"])

    # Dequantize output
    y_pred = y_pred_quantized.astype(np.float32)
    output_f32 = np.round((y_pred - output_zero_point) * output_scale)

    # Print evaluation metrics
    confusion_matrix = np.zeros((2, 2))
    for i in range(len(output_f32)):
        confusion_matrix[int(y_test[i]), int(output_f32[i, 0])] += 1
    print("True positives:     ", int(confusion_matrix[1, 1]))
    print("True negatives:     ", int(confusion_matrix[0, 0]))
    print("False positives:    ", int(confusion_matrix[0, 1]))
    print("False negatives:    ", int(confusion_matrix[1, 0]))


if __name__ == "__main__":
    # Preprocess and load data
    x_train, y_train, x_val, y_val, x_test, y_test = preprocess_and_load_data()

    # Train model
    model = train_model(x_train, y_train, x_val, y_val)

    # Evaluate model
    evaluate_model(model, x_val, y_val, x_test, y_test)

    # Save TFLite model
    tflite_model = export_model_to_tflite(model, x_train)

    # Evaluate TFLite model
    evaluate_tflite_model(tflite_model, x_test, y_test)

    # Generate test case
    generate_test_case()

    print("Done.")
