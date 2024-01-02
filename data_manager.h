#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include "Training_data.h"


// Function prototypes

// Function to save training data to a file (to not have to re-train every time)
void save_training_data(const Training_data* data, const std::string& filename);

// Function to load training data from a file (to not have to re-train every time)
Training_data* load_training_data(const std::string& filename);

// Check if the training data file is available
bool is_training_data_available(const std::string& filename);
