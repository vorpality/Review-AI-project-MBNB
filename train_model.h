#pragma once

#include "Training_data.h"
#include "text_processing.h"
#include <filesystem>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <random>
#include <thread>
#include <mutex>

//Training File skip ratio (0.7 will skip roughly 70% of the training data)
const float SKIPPAGE = 0.9f;

//words per review to be omitted, less than Pk, more than Pn
const float PK = 0.1;
const float PN = 3;


	// Function prototypes

double calculate_entropy(int positive_count, int negative_count);

int calculate_unique_keys(std::map<std::string, int>* map1, std::map<std::string, int>* map2);

int create_map(std::filesystem::path directory_path, std::map<std::string, int>* word_frequency_map);

void process_files(const std::vector<std::filesystem::path>& files, int start, int end, std::map<std::string, int>* word_frequency_map);

Training_data* train(std::filesystem::path dir);
