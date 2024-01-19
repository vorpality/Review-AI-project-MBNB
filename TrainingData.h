#pragma once
#include <filesystem>
#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include <thread>
#include <mutex>
#include "config.h"


namespace fs = std::filesystem;

extern float SHED_RATIO;
extern float SKIPPAGE;

class TrainingData {
private:

    int m_max_vector_size;
    std::vector<std::filesystem::path> m_positive_train_files;
    std::vector<std::filesystem::path> m_negative_train_files;
    std::unordered_map<std::string, int> m_word_index_guide;
    std::vector<float> m_positive_probability_vector;
    std::vector<float> m_negative_probability_vector;
    int m_positive_file_count;
    int m_negative_file_count;
    int m_total;
    double m_entropy;
    std::vector<std::pair<double, std::string>> m_information_gain;

    int calculate_unique_keys(std::map<std::string, int> map1, std::map<std::string, int> map2);

    void initialize_probabilities(
        const std::map<std::string, int>& positive_word_counts, 
        const std::map<std::string, int>& negative_word_counts
    );

    double calculate_entropy(int positive_count, int negative_count);

    std::vector<std::pair<double, std::string>> calculate_information_gain(
        const std::map<std::string, int>& positive_word_counts,
        const std::map<std::string, int>& negative_word_counts
    );

    void sort_and_shed_results(float ratio);

    void shed_results(float ratio);

public:

    TrainingData(
        const std::map<std::string, int>& positive_word_counts,
        const std::map<std::string, int>& negative_word_counts,
        std::vector<std::filesystem::path> positive_files,
        std::vector<std::filesystem::path> negative_files
    );
    ~TrainingData() = default;

    int get_index(std::string word) { return m_word_index_guide[word]; }
    std::unordered_map<std::string, int>& get_guide() { return m_word_index_guide; };
    double get_probability_positive(int index) { return m_positive_probability_vector[index]; };
    double get_probability_negative(int index) { return m_negative_probability_vector[index]; };
    double get_entropy() { return m_entropy; };
    int get_total_files() {return m_total; };
    int get_file_count_positive(){ return m_positive_file_count; };
    int get_file_count_negative(){ return m_negative_file_count; };
    std::vector<std::filesystem::path>get_positive_train_files() { return m_positive_train_files; };
    std::vector<std::filesystem::path>get_negative_train_files() { return m_negative_train_files; };


};
