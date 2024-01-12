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


	// Function prototypes

int create_map(std::filesystem::path directory_path, std::map<std::string, int>& word_frequency_map, float skippage);

void process_files(const std::vector<std::filesystem::path>& files, int start, int end, std::map<std::string, int>& word_frequency_map);

TrainingData train(std::filesystem::path dir, float skippage);
