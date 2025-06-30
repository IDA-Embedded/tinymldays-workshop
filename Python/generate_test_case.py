import json
import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from preprocess import preprocess_audio, WINDOW_SIZE, WINDOW_STRIDE, FRAME_SIZE, FRAME_STRIDE, SPECTRUM_SIZE


RECORDING_ID = 'esp_noise_30cm_1'  # Change this to the desired recording ID
WINDOW_INDEX = 20  # Change this to the desired window index (0-based)


def generate_test_case():
    """
    Loads an audio file from the Data directory, cuts out a second of audio, and saves it as a test case in both JSON and C++.
    """
    # Load the audio file
    sample_rate, audio_data = wavfile.read('../Data/audio_esp_noise_30cm_1.wav')

    # Cut out audio for exactly one window
    start_index = WINDOW_INDEX * WINDOW_STRIDE * FRAME_STRIDE
    audio_sample = audio_data[start_index:start_index + WINDOW_SIZE * FRAME_SIZE + 1]

    # Preprocess the same wave file to get the preprocessed window
    x_test = preprocess_audio(audio_sample)
    if x_test.shape[0] != WINDOW_SIZE or x_test.shape[1] != SPECTRUM_SIZE:
        raise ValueError(f'Expected preprocessed data shape ({WINDOW_SIZE}, {SPECTRUM_SIZE}), but got {x_test.shape}')

    # Load interpreter and get quantization parameters
    interpreter = tf.lite.Interpreter(model_path='gen/model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zero_point = input_details[0]["quantization"]
    output_scale, output_zero_point = output_details[0]["quantization"]

    # Quantize it
    x_test_quantized = x_test / input_scale + input_zero_point
    x_test_quantized = np.clip(x_test_quantized, -128, 127)
    x_test_quantized_int = x_test_quantized.astype(np.int8)

    # Predict with the quantized input
    interpreter.set_tensor(input_details[0]["index"], x_test_quantized_int.reshape(1, WINDOW_SIZE, SPECTRUM_SIZE))
    interpreter.invoke()
    y_pred_quantized = int(interpreter.get_tensor(output_details[0]["index"])[0, 0])

    # Dequantize output
    y_pred = (float(y_pred_quantized) - output_zero_point) * output_scale

    # Save the test case in JSON format
    test_case_dict = {
        "raw": audio_sample.tolist(),
        "preprocessed": x_test.tolist(),
        "quantized": x_test_quantized_int.tolist(),
        "prediction_quantized": y_pred_quantized,
        "prediction": y_pred
    }
    with open('gen/test_case.json', 'w') as json_file:
        json.dump(test_case_dict, json_file)

    # Save the test case in C++ format
    with open('../ESP32/main/test_case.h', 'w') as cpp_file:
        cpp_file.write('#ifndef TEST_CASE_H\n')
        cpp_file.write('#define TEST_CASE_H\n\n')

        cpp_file.write('#include <stdint.h>\n\n')

        cpp_file.write('#define TEST_LENGTH {}\n'.format(len(audio_sample)))

        cpp_file.write('const int32_t raw_audio[TEST_LENGTH] = {\n')
        cpp_file.write(', '.join(map(str, audio_sample)))
        cpp_file.write('\n};\n\n')

        cpp_file.write('const float test_x[{}] = {{\n'.format(WINDOW_SIZE * SPECTRUM_SIZE))
        for row in x_test:
            cpp_file.write('    ' + ', '.join(map(str, row)) + ',\n')
        cpp_file.write('};\n\n')

        cpp_file.write('const int8_t test_xq[{}] = {{\n'.format(WINDOW_SIZE * SPECTRUM_SIZE))
        for row in x_test_quantized_int:
            cpp_file.write('    ' + ', '.join(map(str, row)) + ',\n')
        cpp_file.write('};\n\n')

        cpp_file.write('const float test_prediction = {};\n\n'.format(y_pred))

        cpp_file.write('#endif // TEST_CASE_H\n')


if __name__ == "__main__":
    # Generate test case
    generate_test_case()
