#include "data_manager.h"



struct Training_data;

// Function to save training data to a file (to not have to re-train every time)
void save_training_data(const Training_data* data, const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);
    if (ofs.is_open()) {
		
        // Serialize guide
        size_t guideSize = data->word_index_guide->size();
        ofs.write(reinterpret_cast<const char*>(&guideSize), sizeof(size_t));
        for (const auto& pair : *(data->word_index_guide)) {
            size_t keySize = pair.first.size();
            ofs.write(reinterpret_cast<const char*>(&keySize), sizeof(size_t));
            ofs.write(pair.first.c_str(), keySize);
            ofs.write(reinterpret_cast<const char*>(&(pair.second)), sizeof(int));
        }

        // Serialize positive and negative vectors
        size_t vectorSize = data->positive_probability_vector->size();
        ofs.write(reinterpret_cast<const char*>(&vectorSize), sizeof(size_t));
        for (size_t i = 0; i < vectorSize; ++i) {
            float val = (*data->positive_probability_vector)[i];
            ofs.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }

        vectorSize = data->negative_probability_vector->size();
        ofs.write(reinterpret_cast<const char*>(&vectorSize), sizeof(size_t));
        for (a i = 0; i < vectorSize; ++i) {
            float val = (*data->negative_probability_vector)[i];
            ofs.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
        ofs.write(reinterpret_cast<const char*>(&(data->positive_file_count)), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&(data->negative_file_count)), sizeof(int));

        ofs.close();
    }
}

// Function to load training data from a file (to not have to re-train every time)
Training_data* load_training_data(const std::string& filename) {
    Training_data* data = new Training_data();
    std::ifstream ifs(filename, std::ios::binary);
    if (ifs.is_open()) {
        // Deserialize guide
        size_t guideSize;
        ifs.read(reinterpret_cast<char*>(&guideSize), sizeof(size_t));
        data->word_index_guide = new std::unordered_map<std::string, int>();
        for (size_t i = 0; i < guideSize; ++i) {
            size_t keySize;
            ifs.read(reinterpret_cast<char*>(&keySize), sizeof(size_t));
            std::string key(keySize, ' ');
            ifs.read(reinterpret_cast<char*>(&key[0]), keySize);
            int value;
            ifs.read(reinterpret_cast<char*>(&value), sizeof(int));
            (*(data->word_index_guide))[key] = value;
        }

        // Deserialize positive and negative vectors
        size_t vectorSize;
        ifs.read(reinterpret_cast<char*>(&vectorSize), sizeof(size_t));
        data->positive_probability_vector = new std::vector<float>(vectorSize);
        for (size_t i = 0; i < vectorSize; ++i) {
            float val;
            ifs.read(reinterpret_cast<char*>(&val), sizeof(float));
            (*(data->positive_probability_vector))[i] = val;
        }

        ifs.read(reinterpret_cast<char*>(&vectorSize), sizeof(size_t));
        data->negative_probability_vector = new std::vector<float>(vectorSize);
        for (size_t i = 0; i < vectorSize; ++i) {
            float val;
            ifs.read(reinterpret_cast<char*>(&val), sizeof(float));
            (*(data->negative_probability_vector))[i] = val;
        }

        // Deserialize file counts
        ifs.read(reinterpret_cast<char*>(&(data->positive_file_count)), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&(data->negative_file_count)), sizeof(int));

        ifs.close();
    }
    return data;
}


// Check if the training data file is available
bool is_training_data_available(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}
