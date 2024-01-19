#include "test_data.h"

std::pair<int, int> evaluate_dir_reviews(TrainingData& model, std::filesystem::path review_directory, std::string flag) {
    std::vector<std::filesystem::path> all_review_paths;
    std::string flag_string = (flag == "pos") ? "Positive" : "Negative";
    std::string directory_to_print = (review_directory == "") ? flag_string + " training data" : review_directory.string();
    std::cout << "\nTesting data against : " << directory_to_print << std::endl;
    if (flag == "pos") {
        all_review_paths = model.get_positive_train_files();
    }
    else if (flag == "neg") {
        all_review_paths = model.get_negative_train_files();
    }
    else {
        for (auto const& review_path : std::filesystem::directory_iterator(review_directory)) {
            all_review_paths.push_back(review_path);
        }
        std::shuffle(all_review_paths.begin(), all_review_paths.end(), std::default_random_engine(std::random_device{}()));
        all_review_paths.resize(FILE_CAP);
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
    //Tracker thread 
    threads_vector.emplace_back(track_and_process, std::ref(all_review_paths), 0, reviews_per_thread, std::ref(model), std::ref(positives), std::ref(negatives), std::ref(result_mutex));


    //Remaining threads

    for (int i = 1; i < total_threads; ++i) {
        int start = i * reviews_per_thread;
        int end = (i == total_threads - 1) ? all_review_paths.size() : (i + 1) * reviews_per_thread;
        threads_vector.emplace_back(process_reviews, std::ref(all_review_paths), start, end, std::ref(model), std::ref(positives), std::ref(negatives), std::ref(result_mutex));
    }

    //Join
    for (std::thread& th : threads_vector) {
        if (th.joinable()) {
            th.join();
        }
    }
    return std::pair<int, int>(positives, negatives);
}

void process_reviews(const std::vector<std::filesystem::path>& review_paths, int local_start, int local_end,TrainingData& model, int& positives, int& negatives, std::mutex& result_mutex) {

    for (int index = local_start; index < local_end; index++) {
        
        std::vector<int> review = file_to_vector(review_paths[index], model.get_guide());
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

 void track_and_process(const std::vector<std::filesystem::path>& review_paths, int local_start, int local_end, TrainingData& model, int& positives, int& negatives, std::mutex& result_mutex) {
    for (int index = local_start; index < local_end; index++) {

        std::vector<int> review = file_to_vector(review_paths[index], model.get_guide());
        std::pair<double, double> result = test_vector(review, model);
        // Lock the mutex before accessing the shared result variable
        std::lock_guard<std::mutex> guard(result_mutex);
        if (result.first > result.second) {
            positives++;
        }
        else {
            negatives++;
        }
        std::cout << "\r" << completion_bar(index, local_end);
    }
    std::cout << std::endl;
}



std::pair<double,double> test_vector(std::vector<int> review_vector, TrainingData& model) {

    float Pc0 = (float)model.get_file_count_negative() / (float)(model.get_total_files()); //P(c=0)
    float Pc1 = (float)model.get_file_count_positive() / (float)(model.get_total_files()); //P(c=1)


    long double Px1 = log(Pc1);
    long double Px0 = log(Pc0);

    for (int index = 0; index < review_vector.size(); ++index) {
        int word_exists = review_vector[index];

        //tc = 0 (negative review)
        //tc = 1 (positive review)

        float tc0 = model.get_probability_negative(index);
        float tc1 = model.get_probability_positive(index);


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
