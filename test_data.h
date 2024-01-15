#pragma once

#include "TrainingData.h"
#include "text_processing.h"
#include <vector>
#include <thread>
#include <mutex>
#include <random>
#include <algorithm>
#include "common_helpers.h"


// Function prototypes

void process_reviews(const std::vector<std::filesystem::path>& review_paths, int start, int end, TrainingData& model, int& positives, int& negatives, std::mutex& result_mutex);

void track_and_process(const std::vector<std::filesystem::path>& review_paths, int start, int end, TrainingData& model, int& positives, int& negatives, std::mutex& result_mutex);

std::pair<int,int> evaluate_dir_reviews(TrainingData& model, std::filesystem::path review_directory, std::string flag = "");

std::pair<double, double> test_vector(std::vector<int> review_vector, TrainingData& model);
