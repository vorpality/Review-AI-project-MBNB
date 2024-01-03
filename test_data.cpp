#include "test_data.h"
#include <iostream>
void process_reviews(const std::vector<std::filesystem::path>& review_paths, int local_start, int local_end, Training_data* model, int& positives, int& negatives, std::mutex& result_mutex) {
    for (int index = local_start; index < local_end; index++) {
        
        std::vector<int> review = file_to_vector(review_paths[index], model->word_index_guide);
        std::pair<double,double> result = test_vector(review, model);
        // Lock the mutex before accessing the shared result variable
        std::lock_guard<std::mutex> guard(result_mutex);
        if (result.first > result.second) {
            positives++;
        }
        else {
            negatives++;
        }
    }
}

std::pair<int,int> evaluate_dir_reviews(std::filesystem::path review_directory, Training_data* model){
    std::vector<std::filesystem::path> all_review_paths;

    for (auto const& review_path : std::filesystem::directory_iterator(review_directory)) {
        all_review_paths.push_back(review_path);
    }

    //Calculate available cpu threads
    const int total_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads_vector;
    int reviews_per_thread = all_review_paths.size() / total_threads;

    //shared variables for results
    int positives = 0;
    int negatives = 0;
    
    //Launch threads
    std::mutex result_mutex;
    for (int i = 0; i < total_threads; ++i) {
        int start = i * reviews_per_thread;
        int end = (i == total_threads - 1) ? all_review_paths.size() : (i + 1) * reviews_per_thread;
        threads_vector.emplace_back(process_reviews, std::ref(all_review_paths), start, end, model, std::ref(positives), std::ref(negatives), std::ref(result_mutex));
    }

    //Join
    for (std::thread& th : threads_vector) {
        if (th.joinable()) {
            th.join();
        }
    }
    return std::pair<int, int>(positives, negatives);
}

std::pair<double,double> test_vector(std::vector<int> review_vector, Training_data* model) {

    float Pc0 = (float)model->negative_file_count / (float)(model->positive_file_count + model->negative_file_count); //P(c=0)
    float Pc1 = (float)model->positive_file_count / (float)(model->positive_file_count + model->negative_file_count); //P(c=1)


    long double Px1 = log(Pc1);
    long double Px0 = log(Pc0);

    for (int index = 0; index < review_vector.size(); ++index) {
        int word_exists = review_vector[index];

        //tc = 0 (negative review)
        //tc = 1 (positive review)

        float tc0 = (*model->negative_probability_vector)[index];
        float tc1 = (*model->positive_probability_vector)[index];


        // Print statements for debugging
        //std::cout << "Word index: " << index << " tc1: " << tc1 << " tc0: " << tc0 << std::endl;

        //P(t | c)^x * ( 1-P(t|c) )^(1-x) calculation in log form to avoid underflow

        Px1 += (word_exists * log(tc1)) + ((1 - word_exists) * log(1 - tc1));
        Px0 += (word_exists * log(tc0)) + ((1 - word_exists) * log(1 - tc0));

        //result is positive if Px1 > Px0 
        //negative if Px0 > Px1
    }
    return std::pair<double, double>(Px1, Px0);
}
