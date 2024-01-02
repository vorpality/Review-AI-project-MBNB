#pragma once

#include "Training_data.h"
#include "text_processing.h"
#include <vector>

// Function prototypes

float test_vector(std::vector<int> review_to_test, Training_data* model);
