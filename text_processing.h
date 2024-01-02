#pragma once
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <filesystem>
#include <fstream>
#include <regex>

//Preprocess control.


//words of, or less letter to be omitted
const int MINIMUM_LETTERS = 3;





// Function prototypes

// Removes non-alphabetic characters and converts to lowercase
std::string remove_debris(std::string word);

// Controls word frequency based on PK and PN
void word_frequency_control(int n, int k, std::map<std::string, int>* word_frequency);

// Tokenizes words and builds word frequency dictionary
void add_file_to_map(std::filesystem::path file_path, std::map<std::string, int>* word_frequency);

// Converts file words to a vector based on the guide
std::vector<int> file_to_vector(std::filesystem::path file_path, std::unordered_map<std::string, int>* guide);
