#pragma once

#include "Training_data.h"
#include "text_processing.h"
#include <vector>
#include <thread>
#include <mutex>

// Function prototypes

void process_reviews(const std::vector<std::filesystem::path>& review_paths, int start, int end, Training_data* model, int& positives, int& negatives, std::mutex& result_mutex);

std::pair<int,int> evaluate_dir_reviews(std::filesystem::path review_directory, Training_data* model);

std::pair<double, double> test_vector(std::vector<int> review_vector, Training_data* model);
