#include "train_model.h"

//Helper functions 

//calculates unique keys in 2 maps
int calculate_unique_keys(std::map<std::string, int>* map1, std::map<std::string, int>* map2) {
    std::map<std::string, int> size_counter;

    size_counter.insert(map1->begin(), map1->end());
    size_counter.insert(map2->begin(), map2->end());

    return size_counter.size() + 1;
}

double calculate_entropy(int positive_count, int negative_count) {
    double negative_proportion = (float)negative_count / (positive_count)+(negative_count);
    double positive_proportion = (float)positive_count / (positive_count)+(negative_count);
    double entropy = -(negative_proportion * log2(negative_proportion) + positive_proportion * log2(positive_proportion));
    return entropy;
}

//creates a map[key,value] with [string:word,int count] for every file in directory_path. returns the amount of files parsed
int create_map(std::filesystem::path directory_path, std::map<std::string, int>* word_frequency_map) {

    std::vector<std::filesystem::path> all_files;

    //rng for file-number control
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (const auto& file : std::filesystem::directory_iterator(directory_path)) {
        if (dis(gen) < SKIPPAGE) {
            all_files.push_back(file);
        }
    }

    //available cpu threads

    const int total_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads_vector;
    int files_per_thread = all_files.size() / total_threads;


    //Launch threads
    for (int i = 0; i < total_threads; ++i) {
        int start = i * files_per_thread;
        int end = (i == total_threads - 1) ? all_files.size() : (i + 1) * files_per_thread;
        threads_vector.emplace_back(process_files, std::ref(all_files), start, end, word_frequency_map);
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
void process_files(const std::vector<std::filesystem::path>& local_files, int local_start, int local_end, std::map<std::string, int>* word_frequency_map) {

    for (int index = local_start; index < local_end; index++) {

        std::map<std::string, int> local_map;
        add_file_to_map(local_files[index], &local_map);

        // Lock the mutex before accessing the shared map
        std::lock_guard<std::mutex> guard(map_mutex);
        for (const auto& [word, count] : local_map) {
            (*word_frequency_map)[word] += count;
        }
    }
}

// Function to train the model
Training_data* train(std::filesystem::path dir) {
    std::map<std::string, int>* negative_word_counts = new std::map<std::string, int>;
    std::map<std::string, int>* positive_word_counts = new std::map<std::string, int>;
    std::cout << "\nTraining model.." << std::endl;
    Training_data* data = new Training_data();


    std::filesystem::path negative = dir / "neg";
    std::filesystem::path positive = dir / "pos";

    //calculating word frequency for positive/negative files, used to calculate respective word probabilities.

    int negative_file_count = create_map(negative, negative_word_counts);
    int positive_file_count = create_map(positive, positive_word_counts);

    word_frequency_control((int)PN * negative_file_count, (int)PK * negative_file_count, negative_word_counts);
    word_frequency_control((int)PN * positive_file_count, (int)PK * positive_file_count, positive_word_counts);
   
    max_vector_size = calculate_unique_keys(positive_word_counts, negative_word_counts);
    
    /* transforming dictionaries in indexed vectors for better alignment during testing.
        the "guide" map basically contains every word in positive/negative maps
        as key and an int (index) as value which basically helps creating vectors where
        words are aligned the same way every time.
    */

    std::unordered_map<std::string, int>* word_index_guide = new std::unordered_map<std::string, int>;

    //guide line vectors with ordered attribute probabilities for both negative and positive maps
    // P(C=1|X=1) and P(C=0|X=1)
    std::vector<float>* positive_word_probabilities = new std::vector<float>(max_vector_size);
    std::vector<float>* negative_word_probabilities = new std::vector<float>(max_vector_size);

    int index = 1;
    for (auto element = positive_word_counts->begin(); element != positive_word_counts->end(); element++) {

        std::string word = element->first;
        int word_count = element->second;

        //creating P vector (Respective Probabilities on each index) includes laplace smoothing 
        (*positive_word_probabilities)[index] = (float)(word_count + 1) / (float)(positive_file_count + 2);
        (*negative_word_probabilities)[index] = 1 / (float)(negative_file_count + 2);
        (*word_index_guide)[word] = index;

        index++;
    }

    for (auto element = negative_word_counts->begin(); element != negative_word_counts->end(); element++) {
        std::string word = element->first;
        int word_count = element->second;

        //if word already exists in guide, add it to the respective place
        if (word_index_guide->count(word) > 0) {
            int existing_index = (*word_index_guide)[word];
            (*negative_word_probabilities)[existing_index] = (float)(word_count + 1 / (float)(negative_file_count +  2));
        }
        //adding new words
        else {
            (*negative_word_probabilities)[index] = (float)(word_count + 1) / (float)(negative_file_count + 2);
            (*positive_word_probabilities)[index] = 1 / (float)(positive_file_count + 2);;
            (*word_index_guide)[word] = index;
            index++;
        }
    }
    delete positive_word_counts;
    delete negative_word_counts;

    data->entropy = calculate_entropy(positive_file_count, negative_file_count);
    data->word_index_guide = word_index_guide;
    data->positive_probability_vector = positive_word_probabilities;
    data->negative_probability_vector = negative_word_probabilities;
    data->positive_file_count = positive_file_count;
    data->negative_file_count = negative_file_count;
    return data;
}

