#pragma once

#include "Training_data.h"
#include "text_processing.h"
#include <filesystem>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <random>

//Training File skip ratio (0.7 will skip roughly 70% of the training data)
const float SKIPPAGE = 0.9f;

//words per review to be omitted, less than Pk, more than Pn
const float PK = 0.05;
const float PN = 1;


int max_vector_size;

	// Function prototypes

int create_map(std::filesystem::path directory_path, std::map<std::string, int>* word_frequency_map);

int calculate_unique_keys(std::map<std::string, int>* map1, std::map<std::string, int>* map2);

Training_data* train(std::filesystem::path dir);
