#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <filesystem>
// Configuration for MBNB model training

// Flags (saving and loading dont work currently)
inline bool FULL_TRAIN = false;
inline bool SAVE_MODELS  = false;
inline bool LOAD_MODELS = false;

// Skip Ratios
inline float PK = 0.01f; // Skip ratio of rare words
inline float PK_STEP = 0.02f; //Step for PK during full training
inline float PN = 1.11f;  // Skip ratio of common words
inline float PN_STEP = 0.05f; //Step for PN during full training

// Training Parameters
inline int STARTING_FILES = 50;    // Percentage of files for the first model
inline int MODELS_TO_BE_TRAINED = 20;     // Number of models to be trained
inline int MODEL_FILES_INCREMENT = 50; // Increment for training data in each model

// Directory
inline std::filesystem::path ASSETS_DIR = "assets";
inline std::filesystem::path OUTPUT_DIR = ASSETS_DIR / "output";
inline std::filesystem::path INCLUDE_DIR = ASSETS_DIR / "includes";

inline std::filesystem::path DATASET_DIR = INCLUDE_DIR / "aclImdb"; // Directory to load dataset
inline std::filesystem::path LOAD_DIR = INCLUDE_DIR / "save/model"; // Directory to load/save models
inline std::filesystem::path STOPWORDS_TXT = INCLUDE_DIR / "stopwords.txt"; // Stopwords.txt location
inline std::filesystem::path PY_SCRIPT_DIR = ASSETS_DIR / "py"; //Print python script location
inline std::filesystem::path PLOTTING_SCRIPT = PY_SCRIPT_DIR / "plotting.py";

inline std::filesystem::path FROM_SCRIPT_TO = "../.."/ OUTPUT_DIR;
inline std::string TRAIN_TXT_NAME = "training_data_accuracy.txt";
inline std::string TEST_TXT_NAME = "test_data_accuracy.txt";
// Maximum file processing cap during testing
inline int FILE_CAP = 300;

// Information Gain
inline float SHED_RATIO = 0.15f; // Percentage of features with the highest information gain to be kept

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


// Other global variables 

inline std::unordered_map<std::string, bool> STOPWORDS;

