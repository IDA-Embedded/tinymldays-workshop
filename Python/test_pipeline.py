import json
import numpy as np
import tensorflow as tf
from preprocess import preprocess_audio


def test_pipeline():
    """
    Loads and tests a test case from test_case.json.
    """
    # Load the test case
    with open('gen/test_case.json', 'r') as f:
        test_case_dict = json.load(f)
    audio_sample = np.array(test_case_dict["raw"], dtype=np.float32)

    # Load interpreter and get quantization parameters
    interpreter = tf.lite.Interpreter(model_path='gen/model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zero_point = input_details[0]["quantization"]
    output_scale, output_zero_point = output_details[0]["quantization"]

    # Preprocess
    x_test = preprocess_audio(audio_sample)
    window_size = x_test.shape[0]
    spectrum_size = x_test.shape[1]
    # And check it matches the expected preprocessed matrix
    x_test_expected = np.array(test_case_dict["preprocessed"], dtype=np.float32)
    assert x_test.shape == (window_size, spectrum_size), f'Expected shape ({window_size}, {spectrum_size}), but got {x_test.shape}'
    assert np.allclose(x_test, x_test_expected, atol=1e-5), "Preprocessed data does not match expected values."

    # Quantize
    x_test_quantized = x_test / input_scale + input_zero_point
    x_test_quantized = np.clip(x_test_quantized, -128, 127)
    x_test_quantized_int = x_test_quantized.astype(np.int8)
    # And check it matches the expected quantized matrix
    x_test_quantized_expected = np.array(test_case_dict["quantized"], dtype=np.int8)
    assert np.array_equal(x_test_quantized_int, x_test_quantized_expected), "Quantized data does not match expected values."

    # Predict
    interpreter.set_tensor(input_details[0]["index"], x_test_quantized_int.reshape(1, window_size, spectrum_size))
    interpreter.invoke()
    y_pred_quantized = interpreter.get_tensor(output_details[0]["index"])
    # Dequantize output
    y_pred = y_pred_quantized.astype(np.float32)
    y_pred = (y_pred - output_zero_point) * output_scale
    # And check it matches the expected prediction
    y_pred_expected = test_case_dict["prediction"]
    assert np.isclose(y_pred, y_pred_expected, atol=1e-5), "Prediction does not match expected value."
