#pragma once

#include "TrainingData.h"
#include "text_processing.h"
#include <vector>
#include <thread>
#include <mutex>
#include <random>
#include <algorithm>
#include "common_helpers.h"

//Evaluates all reviews within review_directory via usage of training model given. 
//if flag paramated given is "pos" or "neg" it will ignore the review directory given and will evaluate the relevant training data instead.
//As with training, it will launch threads to do the evaluation, splitting the reviews in {available_threads} vectors.
//The first thread launched is also a tracking thread.
std::pair<int, int> evaluate_dir_reviews(TrainingData& model, std::filesystem::path review_directory, std::string flag = "");

//Thread function that evaluates reviews within given sub-vector
void process_reviews(const std::vector<std::filesystem::path>& review_paths, int start, int end, TrainingData& model, int& positives, int& negatives, std::mutex& result_mutex);

//Tracker thread for console output
void track_and_process(const std::vector<std::filesystem::path>& review_paths, int start, int end, TrainingData& model, int& positives, int& negatives, std::mutex& result_mutex);

//The function that tests a review against the training vectors
std::pair<double, double> test_vector(std::vector<int> review_vector, TrainingData& model);
