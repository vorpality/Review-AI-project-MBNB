#pragma once

#include <iostream>
#include <string>

// Configuration for MBNB model training

// Flags
inline bool FULL_TRAIN = false;
inline bool SAVE_MODELS  = false;
inline bool LOAD_MODELS = false;

// Skip Ratios
inline float PK = 0.01f; // Skip ratio of rare words
inline float PK_STEP = 0.02f; //Step for PK during full training
inline float PN = 1.11f;  // Skip ratio of common words
inline float PN_STEP = 0.5f; //Step for PN during full training

// Training Parameters
inline int STARTING_FILES = 500;    // Percentage of files for the first model
inline int MODELS_TO_BE_TRAINED = 1;     // Number of models to be trained
inline int MODEL_FILES_INCREMENT = 500; // Increment for training data in each model

// Directory
inline std::string LOAD_DIR = "data/model"; // Directory to load/save models

// Maximum file processing cap during testing
inline int FILE_CAP = 500;

// Information Gain
inline float SHED_RATIO = 0.25f; // Percentage of features with the highest information gain to be kept

// Minimum number of letters for word inclusion
inline int MINIMUM_LETTERS = 3;


/*
    Training Example:
    --STARTING_FILES = 0.04
    --MODELS_TO_BE_TRAINED = 10
    --MODEL_FILES_INCREMENT = 0.1f

    --10 models will be trained.
    --the first model will be trained on 4% of the given data
    --the second model will be trained on 14% of the given data
    --the third model will be trained on 24% of the given data, etc.
*/

inline void check_variables() {
    if (LOAD_MODELS) return;

    if (STARTING_FILES <= 0.0f) {
        std::cout << "The STARTING_FILES variable should be greater than 0.\n";
        return;
    }
    if (MODEL_FILES_INCREMENT < 0) {
        std::cout << "The MODEL_FILES_INCREMENT variable is not greater than 0, defaulting to 1 model training.\n";
        MODELS_TO_BE_TRAINED = 1;
        MODEL_FILES_INCREMENT = 0;
    }
    if (MODELS_TO_BE_TRAINED < 0) {
        std::cout << "The MODELS_TO_BE_TRAINED variable is less than 0, defaulting to 1.\n";
        MODELS_TO_BE_TRAINED = 1;
    }
}
