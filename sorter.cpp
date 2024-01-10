#include "sorter.h"

void swap(int index_a, int index_b, Training_data* data) {
    // Assuming information_gain is a vector of pairs (double, string) and accessible
    auto& information_gain = *(data->information_gain);
    auto& word_index_guide = *(data->word_index_guide);
    auto& positive_probability_vector = *(data->positive_probability_vector);
    auto& negative_probability_vector = *(data->negative_probability_vector);

    // Swap words in the word index guide
    std::string word_a = information_gain[index_a].second;
    std::string word_b = information_gain[index_b].second;

    word_index_guide[word_a] = index_b;
    word_index_guide[word_b] = index_a;

    // Swap the information gain values
    std::pair<double, std::string> ig_temp = information_gain[index_a];
    information_gain[index_a] = information_gain[index_b];
    information_gain[index_b] = ig_temp;

    // Swap the positive probability indices
    float ppi_temp = positive_probability_vector[index_a];
    positive_probability_vector[index_a] = positive_probability_vector[index_b];
    positive_probability_vector[index_b] = ppi_temp;

    // Swap the negative probability indices
    float npi_temp = negative_probability_vector[index_a];
    negative_probability_vector[index_a] = negative_probability_vector[index_b];
    negative_probability_vector[index_b] = npi_temp;
}

void bubble_sort_on_ig(Training_data *data) {
    std::vector<std::pair<double, std::string>>* information_gain = data->information_gain;

    bool swapped;
    do {
        swapped = false;
        for (int i = 1; i < data->information_gain->size(); i++) {
            // Assuming you are sorting in descending order of information gain
            if ((*information_gain)[i - 1].first < (*information_gain)[i].first) {
                // Swap elements at index i-1 and i
                swap(i - 1, i, data);
                swapped = true;
            }
        }
    } while (swapped);
}