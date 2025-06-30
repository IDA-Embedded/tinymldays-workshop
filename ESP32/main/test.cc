#include <math.h>
#include "model.h"
#include "preprocess.h"
#include "test_case.h"
#include "esp_log.h"

extern int8_t* model_quantize(const float *features);
extern bool model_predict(float *prediction);

static const char *TAG_INF = "Test";
#define ASSERT(condition) \
    if (!(condition)) { \
        ESP_LOGE(TAG_INF, "Assertion failed at %s:%d\n", __FILE__, __LINE__); \
        while (1); \
    }

void test_pipeline()
{
    // Test preprocess
    static float audio_buffer[FRAME_SIZE];
    static float x[WINDOW_SIZE * SPECTRUM_SIZE];
    float amplitude;
    for (int i = 0; i < TEST_LENGTH - FRAME_SIZE; i += FRAME_SIZE)
    {
        for (int j = 0; j < FRAME_SIZE; j++)
        {
            audio_buffer[j] = raw_audio[i + j];
        }
        preprocess_put_audio(audio_buffer);
        if (preprocess_get_features(x, &amplitude))
        {
            break; // We have a complete feature window
        }
    }
    for (int i = 0; i < WINDOW_SIZE * SPECTRUM_SIZE; i++)
    {
        ASSERT(fabs(test_x[i] - x[i]) < 1e-5);
    }
    
    // Test quantize
    int8_t *xq = model_quantize(x);
    for (int i = 0; i < WINDOW_SIZE * SPECTRUM_SIZE; i++)
    {
        ASSERT(abs(xq[i] - test_xq[i]) <= 1);
    }
    
    // Test prediction
    float prediction;
    ASSERT(model_predict(&prediction));
    ESP_LOGI(TAG_INF, "Expected prediction: %f, Actual prediction: %f", test_prediction, prediction);
    ASSERT(fabs(test_prediction - prediction) < 1e-5); // Assuming the model predicts 0.5 for the test case

    ESP_LOGI(TAG_INF, "*** Pipeline test completed successfully. ***");
}
