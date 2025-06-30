# Workshop Code Review Exercise Solution: Bell Sound Detector for ESP-EYE

This branch of the repository contains the SOLUTION for a code review exercise that is part of the "Why your model fails in the field" workshop of TinyML Days in Aarhus, June 16-17, 2025 (https://events.au.dk/tinymldays). The application is a bell sound event detector with a small neural network trained in Python and deployed on an Espressif ESP-EYE. The application continuously samples the microphone of the ESP-EYE and uses the neural network to detect a bell sound in the audio stream.

## Slides link

https://docs.google.com/presentation/d/19277ulDf3RRV3LonRAghRD_gmoN443knS0zt_qnqLJM

## Exercise solution

To read about the exercise, please switch to the `main` branch.

This branch provides code corrections that address most of the shortcomings in the `main` branch, and the application in this branch can be expected to work much better. The shortcomings of the `main` branch and the improvements in this branch include:

### Poor data quality

* Representation gaps: Missing longer mic distance with ESP32 microphone
    * TODO: Add more such samples
* Representation gaps: Too few difficult negative examples, e.g. metallic exercise equipment noise and voice at the same frequency as the bell
    * TODO: Add more such samples
* Too many redundant "Workout room and music" negative examples
    * Addressed in `Python/preprocess.py` by removing negative windows based on the ratio between positive and negative examples at the end of `_preprocess_recording()`.

### Shortcut learning

* Inadequate preprocessing: Unnecessary frequency resolution (67 bins) for such a simple sound with known frequency profile
    * Addressed in `Python/preprocess.py` by reducing the spectrum to 28 bins, focusing on the known bell sound frequencies and collapsing the other frequencies into combined bins. Implemented with the definitions of `SPECTRUM_SRC` and `SPECTRUM_DST` in the beginning of the file and the `_reduce_spectrum()` function. The reduction parameters are exported to a C header file in `export_model_to_tflite()` in `Python/main.py`.
* Inadequate preprocessing: Uneven feature value distribution
    * Addressed in `Python/preprocess.py` by applying the logarithm (`log1p`) of the FFT output in `_preprocess_recording()`.
* Test set contamination: Random splitting of preprocessed windows
    * Addressed in `Python/preprocess.py` by splitting the data recording-wise in `preprocess_all()` and in `Python/main.py` by shuffling only the training data *after* splitting.
* Lack of model regularization
    * Addressed in `Python/main.py` by adding `Dropout` layers in `train_model()`.

### Pipeline bugs

* No validation of quantized model
    * Addressed in `Python/main.py` with the `evaluate_tflite_model()` function.
* No regression tests with test vectors
    * Addressed with the following files:
       * `Python/generate_test_case.py`, which generates `Python/gen/test_case.json` and `ESP32/main/test_case.h`.
       * `Python/test_pipeline.py`, which runs the training pipeline test case.
       * `ESP32/main/test.cc` and `ESP32/main/test.h`, which runs the inference pipeline test case on the device.
* No audio input sanity checks in the embedded code
    * TODO: Add sanity checks
* No overload check in the embedded code
    * TODO: Add overload checks
* Hard-coded quantization parameters
    * Addressed in `ESP32/main.cc` by using the quantization parameters embedded in the model binary.

## Repository structure

The repository has two projects:

* A Python project in the `Python` folder. This project reads data and labels from the `Data` folder, trains a machine learning model using TensorFlow and generates C code from the resulting model directly into the microcontroller project in the `ESP32` folder.
* A microcontroller project in the `ESP32` folder. This project contains the actual application code.

The repository root also contains the sound file `ding.wav`, which is the bell sound to be detected.

**NOTE**: You must run the Python project before you can build the ESP32 project, because the generated ML model files and the test case (`model.c`, `model.h` and `test_case.h`) are not included in the repository.

### Training data

Training data is located in the `Data` folder. Each recording is represented by two files:

* A .wav file with 16-bit mono audio at 16 kHz. The file name is `audio_`*\<id\>*`.wav`, where *\<id\>* can be any unique string that identifies the recording.
* A .txt file with labels indicating the starting and ending times of each bell sound in the recording. Each row contains a starting time in seconds, the ending time in seconds and a label name, separated by a tab character. This conforms to Audacity's label file format. The file name is `label_`*\<id\>*`.txt`, where *\<id\>* is the recording identifier.

The `Data` folder also contains a CSV file `metadata.csv` with metadata about each recording.

## Python and ESP32 setup (optional)

You don't need to run the training and build the microcontroller project - it's fully possible to understand the solution only by inspecting the data and the code. However, if you do want to train and build, you need to set up a Python environment as well as the ESP32 build tools on your computer.

Many IDEs, including VSCode and PyCharm, have extensions and built-in features for installing both Python and ESP32 build tools. If you use any of these IDEs, it's recommended to use these for installation. The instructions below can be used for a minimal IDE-independent installation to be run via a terminal or command prompt.

**NOTE**: Installing, training, and building the project take considerable time if you don't already have the tools on your computer. Expect it to take at least one hour, even if nothing goes wrong. There will not be time during the workshop to install, train and build, so you will need to do it in advance.

### Linux & Mac

**DISCLAIMER**: I haven't been able to test setup on Linux and Mac. If you experience problems and want to share a solution, please email me at `gustaf@waveworks` (add `.dk`).

#### Python:

1. Install python environment: (Debian package manager in this example. Use yours)
    ```sh
        sudo apt install python3-venv
    ```
2. CD into the Python project directory:
    ```sh
        cd <repository-root>/Python
    ```
3. Create a virtual environment and activate it:
    ```sh
        python3 -m venv .venv && source ./.venv/bin/activate
    ```
4. Upgrade pip and install dependencies:
    ```sh
        pip install --upgrade pip && pip install -r requirements.txt
    ```
5. Run the training script:
    ```sh
        python3 main.py
    ```

#### ESP32:

1. Install ESP-IDF according to the following instructions: https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/linux-macos-setup.html
2. CD into the ESP32 project directory:
    ```sh
        cd <repository-root>/ESP32
    ```
3. Build the project:
    ```sh
        idf.py build
    ```
4. With an ESP-EYE device connected to your computer, flash and run the project:
    ```sh
        idf.py flash monitor
    ```

### Windows

#### Python:

1. Install a Python environment from Microsoft Store: https://apps.microsoft.com/detail/9NRWMJP3717K
2. Open a command prompt and cd into the Python project directory:
    ```sh
        cd <repository-root>\Python
    ```
3. Create a virtual environment and activate it:
    ```sh
        python -m venv .venv && .\.venv\Scripts\activate
    ```
4. Install dependencies:
    ```sh
        pip install -r requirements.txt
    ```
5. Run the training script:
    ```sh
        python main.py
    ```

#### ESP32:

1. Install ESP-IDF according to the following instructions: https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/windows-setup.html
    - When asked for an ESP-IDF installation directory, choose `C:\Users\<your-user-name>\esp\esp-idf`.
    - When asked for a tools installation directory, choose `C:\Users\<your-user-name>\.espressif`.
    - When asked for components, choose at least:
        - Frameworks
        - Command Prompt
        - Espressif (WinUSB support for JTAG)
        - ESP32 chip target
2. Open the ESP-IDF command prompt and cd into the ESP32 project directory:
    ```sh
        cd <repository-root>\ESP32
    ```
3. Build the project:
    ```sh
        idf.py build
    ```
4. With an ESP-EYE device connected to your computer, flash and run the project:
    ```sh
        idf.py flash monitor
    ```

