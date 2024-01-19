#pragma once
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <filesystem>
#include <fstream>
#include <regex>
#include "config.h"


//Preprocess functions

//Loads stopwords provided in STOPWORDS_TXT (found in config.h) so it can ignore them during word selection
void load_stopwords(const std::filesystem::path& filename = STOPWORDS_TXT);

//Removes non-alphabetic characters and converts to lowercase
std::string remove_debris(std::string word);

//Controls word frequency based on PK and PN (which are ratios instead of numbers, so it works the same for all amounts of training data)
void word_frequency_control(int n, int k, std::map<std::string, int>& word_frequency);

//Tokenizes words and builds word frequency dictionary
void add_file_to_map(std::filesystem::path file_path, std::map<std::string, int>& word_frequency);

//Testing functions 
//Converts file words to a vector based on the guide
std::vector<int> file_to_vector(std::filesystem::path file_path, std::unordered_map<std::string, int>& guide);
