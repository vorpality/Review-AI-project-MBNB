#pragma once
#include <vector>
#include <string>
#include <unordered_map>

//swaps two elements in index_a and index_b, across all vectors to keep them aligned
void swap(int index_a, int index_b,
    std::vector<std::pair<double, std::string>>& information_gain,
    std::unordered_map<std::string, int>& word_index_guide,
    std::vector<float>& positive_probability_vector,
    std::vector<float>& negative_probability_vector);

//Manual bubble sorting(because of aligned vectors) information gain vector in descending order, keeps all other vectors aligned
void bubble_sort_on_ig(std::vector<std::pair<double, std::string>>& information_gain,
    std::unordered_map<std::string, int>& word_index_guide,
    std::vector<float>& positive_probability_vector,
    std::vector<float>& negative_probability_vector);
