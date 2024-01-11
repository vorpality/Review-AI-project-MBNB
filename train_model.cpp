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
    double negative_proportion = (float)negative_count + 1 / (positive_count + negative_count + 2);
    double positive_proportion = (float)positive_count + 1 / (positive_count + negative_count + 2);
    double entropy = -(negative_proportion * log2(negative_proportion) + positive_proportion * log2(positive_proportion));
    return entropy;
}

//creates a map[key,value] with [string:word,int count] for every file in directory_path. returns the amount of files parsed
int create_map(std::filesystem::path directory_path, std::map<std::string, int>* word_frequency_map, float skippage) {

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
    std::mutex map_mutex;

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

void shed_results(Training_data* data, float ratio = 0.2) {

    size_t cutoff_index = static_cast<size_t>(ratio * data->information_gain->size());

    data->information_gain->resize(cutoff_index);
    data->positive_probability_vector->resize(cutoff_index);
    data->negative_probability_vector->resize(cutoff_index);

    // New map for updated word index guide
    std::unordered_map<std::string, int> new_word_index_guide;

    // Rebuild the word index guide based on the trimmed information_gain
    for (int i = 0; i < data->information_gain->size(); ++i) {
        const std::string& word = (*data->information_gain)[i].second; // word from the information gain pair
        new_word_index_guide[word] = i; // New index corresponding to its position in the trimmed vector
    }

    // Replace old word index guide with the new one
    *data->word_index_guide = new_word_index_guide;

}


std::vector<std::pair<double,std::string>>* calculate_information_gain(Training_data* data, std::map<std::string, int> *positive_wordmap, std::map<std::string, int>*negative_wordmap) {
    std::unordered_map<std::string, int>* word_index_guide = data->word_index_guide;
    std::vector<std::pair<double, std::string>>* results = new std::vector<std::pair<double, std::string>>(word_index_guide->size());
    int positive_with=0, negative_with=0;
    for (auto& entry : (*word_index_guide)) {
        std::string word = entry.first;
        int vector_index = entry.second;
        int positive_count = data->positive_file_count;
        int negative_count = data->negative_file_count;

        try {
            positive_with = positive_wordmap->at(word)+1;
        }
        catch (...){
            positive_with = 1;
        }        
        try {
            negative_with = negative_wordmap->at(word)+1;
        }
        catch (...){
            negative_with = 1;
        }
        int total_count = positive_count + negative_count;

        int positive_without = positive_count - positive_with;
        int negative_without = negative_count - negative_with;

        double entropy_with = calculate_entropy(positive_with, negative_with);
        double entropy_without = calculate_entropy(positive_without, negative_without);

        double weight_with = (float)(positive_with + negative_with) / (float)total_count;
        double weight_without = (float)(positive_without + negative_without) / (float)total_count;

        double weighted_entropy = (weight_with * entropy_with) + (weight_without * entropy_without);
        double information_gain = data->entropy - weighted_entropy;
        
        (*results)[vector_index] = std::pair<double, std::string>(information_gain, word);
    }    return results;

}


// Function to train the model
Training_data* train(std::filesystem::path dir, float skippage = 0.5f) {
    std::map<std::string, int>* negative_word_counts = new std::map<std::string, int>;
    std::map<std::string, int>* positive_word_counts = new std::map<std::string, int>;
    int max_vector_size;

    Training_data* data = new Training_data();


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
   
    max_vector_size = calculate_unique_keys(positive_word_counts, negative_word_counts)-1;
    
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

    int index = 0;
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
            (*negative_word_probabilities)[existing_index] = (float)(word_count + 1) / (float)(negative_file_count +  2);
        }
        //adding new words
        else {
            (*negative_word_probabilities)[index] = (float)(word_count + 1) / (float)(negative_file_count + 2);
            (*positive_word_probabilities)[index] = 1 / (float)(positive_file_count + 2);;
            (*word_index_guide)[word] = index;
            index++;
        }
    }

    data->entropy = calculate_entropy(positive_file_count, negative_file_count);
    data->word_index_guide = word_index_guide;
    data->positive_probability_vector = positive_word_probabilities;
    data->negative_probability_vector = negative_word_probabilities;
    data->positive_file_count = positive_file_count;
    data->negative_file_count = negative_file_count;

    data->information_gain = calculate_information_gain(data, positive_word_counts, negative_word_counts);
    bubble_sort_on_ig(data);
    shed_results(data);
    delete positive_word_counts;
    delete negative_word_counts;

    return data;
}

