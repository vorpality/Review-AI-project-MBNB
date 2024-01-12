#pragma once
#include <vector>
#include <string>
#include <unordered_map>

void swap(int index_a, int index_b,
    std::vector<std::pair<double, std::string>>& information_gain,
    std::unordered_map<std::string, int>& word_index_guide,
    std::vector<float>& positive_probability_vector,
    std::vector<float>& negative_probability_vector);

void bubble_sort_on_ig(std::vector<std::pair<double, std::string>>& information_gain,
    std::unordered_map<std::string, int>& word_index_guide,
    std::vector<float>& positive_probability_vector,
    std::vector<float>& negative_probability_vector);
