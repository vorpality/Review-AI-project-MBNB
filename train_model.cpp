#include "train_model.h"

//Helper functions 

//creates a map[key,value] with [string:word,int count] for every file in directory_path. returns the amount of files parsed
std::vector<std::filesystem::path> create_map(std::filesystem::path directory_path, std::map<std::string, int>& word_frequency_map, int num_files) {

    std::vector<std::filesystem::path> all_files;
    std::cout << "\nTraining from : " << directory_path.string();


    for (const auto& file : std::filesystem::directory_iterator(directory_path)) {
        all_files.push_back(file);
    }
    std::shuffle(all_files.begin(), all_files.end(), std::default_random_engine(std::random_device{}()));
    all_files.resize(num_files);
    //Calculate available cpu threads

    const int total_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads_vector;
    int files_per_thread = all_files.size() / total_threads;


    //Launch threads

    // Tracking thread

    threads_vector.emplace_back(track_and_process, std::ref(all_files), 0, files_per_thread, std::ref(word_frequency_map));
    //Remaining threads
    for (int i = 1; i < total_threads; ++i) {
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

    return all_files;
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



void track_and_process(const std::vector<std::filesystem::path>& local_files, int local_start, int local_end, std::map<std::string, int>& word_frequency_map) {
    std::mutex map_mutex;
    std::cout << std::endl;
    for (int index = local_start; index < local_end; index++) {

        std::map<std::string, int> local_map;
        add_file_to_map(local_files[index], local_map);

        // Lock the mutex before accessing the shared map
        std::lock_guard<std::mutex> guard(map_mutex);
        for (const auto& [word, count] : local_map) {
            word_frequency_map[word] += count;
        }
        std::cout << "\r" <<  completion_bar(index, local_end);
    }
    std::cout << std::endl;
}

// Function to train and return a model
TrainingData train(std::filesystem::path dir, int num_files = 1500) {
    std::map<std::string, int> negative_word_counts;
    std::map<std::string, int> positive_word_counts;

    std::filesystem::path negative = dir / "neg";
    std::filesystem::path positive = dir / "pos";

    //calculating word frequency for positive/negative files, used to calculate respective word probabilities.
    std::vector<std::filesystem::path> negative_files = create_map(negative, negative_word_counts, num_files);
    std::vector<std::filesystem::path> positive_files = create_map(positive, positive_word_counts, num_files);
    float n_negative = PN * negative_files.size();
    float n_positive = PN * positive_files.size();
    float k_negative = PK * negative_files.size();
    float k_positive = PK * positive_files.size();
    word_frequency_control(n_negative, k_negative, negative_word_counts);
    word_frequency_control(n_positive, k_positive, positive_word_counts);
   
    return TrainingData(
        positive_word_counts, 
        negative_word_counts,
        positive_files,
        negative_files
    );
}

