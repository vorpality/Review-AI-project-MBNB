#pragma once
#include <string>
#include <unordered_map>
#include <vector>

// Definition of Training_data struct
struct Training_data {
    std::unordered_map<std::string, int>* word_index_guide;
    std::vector<float>* positive_probability_vector;
    std::vector<float>* negative_probability_vector;
    int positive_file_count;
    int negative_file_count;
    double entropy;
    std::vector<std::pair<double, std::string>>* information_gain;
};
