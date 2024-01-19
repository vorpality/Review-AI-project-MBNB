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

//adds the unique words of each file in directory_path 
//to a word_frequency_map[key,value] with [string:word,int:count] 
//caps files processes to num_files
std::vector<std::filesystem::path> create_map(std::filesystem::path directory_path, std::map<std::string, int>& word_frequency_map, int num_files);

//Thread function for create_map, works on a smaller part array (from int start to int end) 
void process_files(const std::vector<std::filesystem::path>& files, int start, int end, std::map<std::string, int>& word_frequency_map);

//Tracking thread function, same as process_files() but has console output to for tracking
void track_and_process(const std::vector<std::filesystem::path>& files, int start, int end, std::map<std::string, int>& word_frequency_map);

//Prepares data in directory dir for training, creates the word frequency map and then calls TrainingData() to finish training and returns the resulting model.
//num_files is the cap of files to be used
//uses k and n to cutoff words
TrainingData train(std::filesystem::path dir, int num_files = 1500);
