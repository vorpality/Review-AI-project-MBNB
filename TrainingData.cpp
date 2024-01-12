#pragma once
#include "TrainingData.h"
#include "sorter.h"

#include <random>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <iostream>


TrainingData::TrainingData( 
    const std::map<std::string, int>& positive_word_counts,
    const std::map<std::string, int>& negative_word_counts,
    int positive_file_count, 
    int negative_file_count
    )
    : m_positive_file_count(positive_file_count),
      m_negative_file_count(negative_file_count) 
{
    m_total = m_positive_file_count + m_negative_file_count;

    initialize_probabilities(positive_word_counts, negative_word_counts);
    m_entropy = calculate_entropy(m_positive_file_count, m_negative_file_count);

    m_information_gain = calculate_information_gain(positive_word_counts, negative_word_counts);

    sort_and_shed_results(SHED_RATIO);
}

int TrainingData::calculate_unique_keys(std::map<std::string, int> map1, std::map<std::string, int> map2) {
    std::map<std::string, int> size_counter;

    size_counter.insert(map1.begin(), map1.end());
    size_counter.insert(map2.begin(), map2.end());

    return size_counter.size() + 1;
}

/* transforming dictionaries in indexed vectors for better alignment during testing.
    the "guide" map basically contains every word in positive/negative maps
    as key and an int (index) as value which basically helps creating vectors where
    words are aligned the same way every time.
    P(C=1|X=1) and P(C=0|X=1)
*/

void TrainingData::initialize_probabilities(
    const std::map<std::string, int>& positive_word_counts, 
    const std::map<std::string, int>& negative_word_counts
    ) {
    // Determine the maximum size for the vectors
    m_max_vector_size = calculate_unique_keys(positive_word_counts, negative_word_counts);
    m_positive_probability_vector.resize(m_max_vector_size, 0.0f);
    m_negative_probability_vector.resize(m_max_vector_size, 0.0f);

    int index = 0;
    for (const auto& element : positive_word_counts) {
        const std::string& word = element.first;
        int word_count = element.second;

        // Creating P vector (Respective Probabilities on each index) includes Laplace smoothing
        m_positive_probability_vector[index] = static_cast<float>(word_count + 1) / static_cast<float>(m_positive_file_count + 2);

        // Initialize negative probability as well in case it's not in negative_word_counts
        m_negative_probability_vector[index] = 1.0f / static_cast<float>(m_negative_file_count + 2);

        m_word_index_guide[word] = index;
        index++;
    }

    // Iterate through the negative word counts and update probabilities
    for (const auto& element : negative_word_counts) {
        const std::string& word = element.first;
        int word_count = element.second;

        // Check if the word is already in the word index guide
        if (m_word_index_guide.count(word) > 0) {
            int existing_index = m_word_index_guide[word];
            m_negative_probability_vector[existing_index] = static_cast<float>(word_count + 1) / static_cast<float>(m_negative_file_count + 2);
        }
        else {
            // If the word is new, add it to the vectors
            m_negative_probability_vector[index] = static_cast<float>(word_count + 1) / static_cast<float>(m_negative_file_count + 2);
            m_positive_probability_vector[index] = 1.0f / static_cast<float>(m_positive_file_count + 2);
            m_word_index_guide[word] = index;
            index++;
        }
    }
}

// Calculating entropy based on 
// -( P(C=0) * log2( P(C=0) ) + ( P(C=1) * log2( P(C=1) ) )

double TrainingData::calculate_entropy(int positive_count, int negative_count) {
    double total_count = positive_count + negative_count;
    double negative_proportion = (double)(negative_count + 1) / (total_count + 2);
    double positive_proportion = (double)(positive_count + 1) / (total_count + 2);

    if (negative_proportion == 0) negative_proportion = 1 / (total_count + 2);
    if (positive_proportion == 0) positive_proportion = 1 / (total_count + 2);

    double entropy = -(negative_proportion * log2(negative_proportion) + positive_proportion * log2(positive_proportion));

    return entropy;
}

/* Calculating information gain based on :
    Entropy(T) - 
    {(Tw/T) * Entropy(Tw) + (Tnw) * Entropy(Tnw)}

    T : set of files
    w : word exists
    mw : word doesn't exist
*/
std::vector<std::pair<double, std::string>> TrainingData::calculate_information_gain(
    const std::map<std::string, int>& positive_word_map,
    const std::map<std::string, int>& negative_word_map
    ) {
    std::vector<std::pair<double, std::string>> results = std::vector<std::pair<double, std::string>>(m_word_index_guide.size()+1);
    int positive_with = 0, negative_with = 0;

    for (auto& entry : (m_word_index_guide)) {
        std::string word = entry.first;
        int vector_index = entry.second;

        auto pos_it = positive_word_map.find(word);
        positive_with = (pos_it != positive_word_map.end()) ? pos_it->second + 1 : 1;

        auto neg_it = negative_word_map.find(word);
        negative_with = (neg_it != negative_word_map.end()) ? neg_it->second + 1 : 1;
        int total_count = m_positive_file_count + m_negative_file_count;

        int positive_without = m_positive_file_count - positive_with;
        int negative_without = m_negative_file_count - negative_with;

        double entropy_with = calculate_entropy(positive_with, negative_with);
        double entropy_without = calculate_entropy(positive_without, negative_without);

        double weight_with = (float)(positive_with + negative_with) / (float)total_count;
        double weight_without = (float)(positive_without + negative_without) / (float)total_count;

        double weighted_entropy = (weight_with * entropy_with) + (weight_without * entropy_without);
        double information_gain = m_entropy - weighted_entropy;

        results[vector_index] = std::make_pair(information_gain, word);
    }    return results;

}

void TrainingData::sort_and_shed_results(float ratio = 0.2) {

    bubble_sort_on_ig(m_information_gain, m_word_index_guide,
        m_positive_probability_vector, m_negative_probability_vector);

    shed_results(ratio);
    return;
}

void TrainingData::shed_results(float ratio) {

    size_t cutoff_index = static_cast<size_t>(ratio * m_information_gain.size());

    m_information_gain.resize(cutoff_index);
    m_positive_probability_vector.resize(cutoff_index);
    m_negative_probability_vector.resize(cutoff_index);

    // New map for updated word index guide
    std::unordered_map<std::string, int> new_word_index_guide;

    // Rebuild the word index guide based on the trimmed information_gain
    for (int i = 0; i < m_information_gain.size(); ++i) {
        const std::string& word = (m_information_gain)[i].second; // word from the information gain pair
        new_word_index_guide[word] = i; // New index corresponding to its position in the trimmed vector
    }

    // Replace old word index guide with the new one
    m_word_index_guide = new_word_index_guide;

}


