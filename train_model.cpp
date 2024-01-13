#include "train_model.h"

//Helper functions 

//creates a map[key,value] with [string:word,int count] for every file in directory_path. returns the amount of files parsed
int create_map(std::filesystem::path directory_path, std::map<std::string, int>& word_frequency_map, float skippage) {

    std::vector<std::filesystem::path> all_files;

    //rng for file-number control
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (const auto& file : std::filesystem::directory_iterator(directory_path)) {
        if (dis(gen) < skippage) {
            all_files.push_back(file);
        }
    }

    //Calculate available cpu threads

    const int total_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads_vector;
    int files_per_thread = all_files.size() / total_threads;


    //Launch threads
    for (int i = 0; i < total_threads; ++i) {
        int start = i * files_per_thread;
        int end = (i == total_threads - 1) ? all_files.size() : (i + 1) * files_per_thread;
        threads_vector.emplace_back(process_files, std::ref(all_files), start, end, std::ref(word_frequency_map));
    }

    //Join
    for (std::thread& th : threads_vector) {
        if (th.joinable()) {
            th.join();
        }
    }

    return all_files.size();
}

//function to use multithreading for faster training
void process_files(const std::vector<std::filesystem::path>& local_files, int local_start, int local_end, std::map<std::string, int>& word_frequency_map) {
    std::mutex map_mutex;

    for (int index = local_start; index < local_end; index++) {

        std::map<std::string, int> local_map;
        add_file_to_map(local_files[index], local_map);

        // Lock the mutex before accessing the shared map
        std::lock_guard<std::mutex> guard(map_mutex);
        for (const auto& [word, count] : local_map) {
            word_frequency_map[word] += count;
        }
    }
}


// Function to train and return a model
TrainingData train(std::filesystem::path dir, float skippage = 0.5f) {
    std::map<std::string, int> negative_word_counts;
    std::map<std::string, int> positive_word_counts;

    std::filesystem::path negative = dir / "neg";
    std::filesystem::path positive = dir / "pos";

    //calculating word frequency for positive/negative files, used to calculate respective word probabilities.

    int negative_file_count = create_map(negative, negative_word_counts,skippage);
    int positive_file_count = create_map(positive, positive_word_counts,skippage);
    float n_negative = PN * negative_file_count;
    float n_positive = PN * positive_file_count;
    float k_negative = PK * negative_file_count;
    float k_positive = PK * positive_file_count;
    word_frequency_control(n_negative, k_negative, negative_word_counts);
    word_frequency_control(n_positive, k_positive, positive_word_counts);
   
    return TrainingData(
        positive_word_counts, 
        negative_word_counts,
        positive_file_count, 
        negative_file_count
    );
}

