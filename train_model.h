#pragma once

#include "TrainingData.h"
#include "text_processing.h"
#include <filesystem>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <random>
#include <thread>
#include <mutex>
#include "sorter.h"
#include "common_helpers.h"


	// Function prototypes

std::vector<std::filesystem::path> create_map(std::filesystem::path directory_path, std::map<std::string, int>& word_frequency_map, int num_files);

void process_files(const std::vector<std::filesystem::path>& files, int start, int end, std::map<std::string, int>& word_frequency_map);

void track_and_process(const std::vector<std::filesystem::path>& files, int start, int end, std::map<std::string, int>& word_frequency_map);

TrainingData train(std::filesystem::path dir, int num_files);

std::string completion_bar(int current, int total);
