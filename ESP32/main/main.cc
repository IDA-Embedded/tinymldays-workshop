#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "esp_log.h"

// Include TFLM
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "examples/micro_speech/main/feature_provider.h"

// Project includes
#include "audio.h"
#include "preprocess.h"
#include "model.h"
#include "test.h"
#define MODEL_BINARY model_binary

// Compile time check preprocessor constants
#if SAMPLE_RATE != AUDIO_SAMPLE_RATE
#error "AUDIO_SAMPLE_RATE must be equal to SAMPLE_RATE from model.h"
#endif
#if FRAME_SIZE != AUDIO_BUFFER_SAMPLES
#error "AUDIO_BUFFER_SAMPLES must be equal to FRAME_SIZE from model.h"
#endif

// Tensor arena size, found by trial and error
#define TENSOR_ARENA_SIZE (30 * 1024)

// Static variables
static const tflite::Model *model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
float float_features_buffer[WINDOW_SIZE * SPECTRUM_SIZE];
static TfLiteTensor *input = nullptr;
static TfLiteTensor *output = nullptr;
static const char *TAG_INF = "Inference";

/**
 * @brief Main setup function.
 */
void setup(void)
{
    // Load TFlite model
    model = tflite::GetModel(MODEL_BINARY);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        ESP_LOGE(TAG_INF, "Model schema mismatch!");
        abort();
    }

    // Create an interpreter
    static tflite::MicroMutableOpResolver<6> micro_op_resolver;
    // Conv1D
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddConv2D();
    // MaxPool1D
    micro_op_resolver.AddMaxPool2D();
    // Flatten
    micro_op_resolver.AddExpandDims();
    // Dense with sigmoid activation
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddLogistic();
    static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    // Allocate memory for input and output tensors
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        ESP_LOGE(TAG_INF, "Failed to allocate tensors!");
        abort();
    }

    // Get pointers for input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);

    // Print input and output tensor types and dimensions
    ESP_LOGI(TAG_INF, "Input tensor type: %s, shape: %d, %d, %d",
             TfLiteTypeGetName(input->type), input->dims->data[0], input->dims->data[1], input->dims->data[2]);
    ESP_LOGI(TAG_INF, "Output tensor type: %s, shape: %d",
             TfLiteTypeGetName(output->type), output->dims->data[0]);

    // Initialize audio with gain 16.0 (found experimentally)
    audio_init(16.0f);

    // Initialize preprocessing
    if (!preprocess_init())
    {
        ESP_LOGE(TAG_INF, "Failed to initialize preprocessing!");
        abort();
    }

    // Run test case
    test_pipeline();
}

/**
 *  @brief Quantize the feature matrix from float to int8.
 *  @param features Pointer to the input feature matrix in float format.
 *  @return Pointer to the output feature matrix in int8 format.
 */
int8_t* model_quantize(const float *features)
{
    // Quantize the feature matrix from float to int8
    for (size_t i = 0; i < WINDOW_SIZE * SPECTRUM_SIZE; ++i)
    {
        float val_quant_float = roundf(features[i] / input->params.scale) + input->params.zero_point;
        input->data.int8[i] = static_cast<int8_t>(val_quant_float);
    }

    // Return pointer to the quantized input tensor (used in test function)
    return input->data.int8;
}

/**
 * @brief Run inference on the model, obtain the prediction and dequantize to float.
 * @param prediction Pointer to store the prediction result.
 * @return True if inference was successful, false otherwise.
 */
bool model_predict(float *prediction)
{
    // Run inference
    if (interpreter->Invoke() != kTfLiteOk)
    {
        return false;
    }

    // Dequantize the output from int8 to float
    *prediction = (static_cast<float>(output->data.int8[0]) - output->params.zero_point) * output->params.scale;
    return true;
}

/**
 * @brief Main loop function.
 */
void loop(void)
{
    // Obtain audio frame
    float *audio_frame = audio_read();

    // Put and preprocess audio frame
    preprocess_put_audio(audio_frame);

    // Obtain features and run inference if features are ready
    float amplitude;
    if (preprocess_get_features(float_features_buffer, &amplitude))
    {
        // Quantize the input features from float to int8 into the input tensor
        model_quantize(float_features_buffer);

        // Run inference
        float prediction;
        if (!model_predict(&prediction))
        {
            ESP_LOGE(TAG_INF, "Failed to invoke interpreter!");
        }
        else
        {
            // Print output
            ESP_LOGI(TAG_INF, "Amplitude: %5.0f, Prediction: %.2f", amplitude, prediction);
        }
    }
}

extern "C" void app_main(void)
{
    setup();
    while (true)
    {
        loop();
    }
}
