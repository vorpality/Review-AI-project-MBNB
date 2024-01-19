#include "sorter.h"

void swap(int index_a, int index_b,
    std::vector<std::pair<double, std::string>>& information_gain,
    std::unordered_map<std::string, int>& word_index_guide,
    std::vector<float>& positive_probability_vector,
    std::vector<float>& negative_probability_vector) {

    std::swap(information_gain[index_a], information_gain[index_b]);
	
    word_index_guide[information_gain[index_a].second] = index_a;
    word_index_guide[information_gain[index_b].second] = index_b;
	
    std::swap(positive_probability_vector[index_a], positive_probability_vector[index_b]);
    std::swap(negative_probability_vector[index_a], negative_probability_vector[index_b]);
}

void bubble_sort_on_ig(std::vector<std::pair<double, std::string>>& information_gain,
    std::unordered_map<std::string, int>& word_index_guide,
    std::vector<float>& positive_probability_vector,
    std::vector<float>& negative_probability_vector) {

    bool swapped;
    do {
        swapped = false;
        for (int i = 1; i < information_gain.size(); i++) {
            if (information_gain[i - 1].first < information_gain[i].first) {
                swap(i - 1, i, 
                    information_gain, word_index_guide,
                    positive_probability_vector, negative_probability_vector);
                swapped = true;
            }
        }
    } while (swapped);
}