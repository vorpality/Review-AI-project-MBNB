#include "text_processing.h"

void load_stopwords(const std::filesystem::path& filename)
{
    std::unordered_map<std::string, bool> stopwords;
    std::string word;
    std::ifstream file(STOPWORDS_TXT); 
    if (!file) {
        std::cerr << "Error opening file: " << STOPWORDS_TXT << std::endl;
        return;
    }

    while (file >> word) {
        stopwords[word] = true;
    }

    STOPWORDS = stopwords;
    return;
}

std::string remove_debris(std::string word)
{
    for (int i = 0; i < word.size(); i++) {
        word[i] = tolower(word[i]);
    }
    word = std::regex_replace(word, std::regex("[^a-zA-Z\\s]"), "");
    return word;
}

void word_frequency_control(int n, int k, std::map<std::string, int>& word_frequency) {
    for (auto it = word_frequency.begin(); it != word_frequency.end();) {
        if (it->second >= n || 
            it->second <= k ||
            STOPWORDS.find(it->first) != STOPWORDS.end()
            ) {
            word_frequency.erase(it++);
        }
        else {
            ++it;
        }
    }
}

void add_file_to_map(std::filesystem::path file_path, std::map<std::string, int>& word_frequency) {
    std::unordered_map<std::string, bool> unique_words;
    std::string path_string = file_path.string();
    std::ifstream file(file_path);
    std::string word;
    while (std::getline(file, word, ' ')) {
        // Removes special characters and changes words to all lowercase
        word = remove_debris(word);
        
        //dumps words with less than MINIMUM_LETTERS
        if (word.length() < MINIMUM_LETTERS) {
            continue;
        }

        //Skips multiple occurences of the same word
        if (unique_words.count(word) > 0) {
            continue;
        }
        if (word_frequency.count(word) == 0) {
            word_frequency[word] = 1;

        }
        else {
            word_frequency[word] += 1;
        }
        unique_words[word] = true;
    }
    unique_words.clear();
}

std::vector<int> file_to_vector(std::filesystem::path file_path, std::unordered_map<std::string, int>& guide) {
    std::string path_string = file_path.string();
    std::ifstream file(file_path);
    std::string word;
    std::vector<int> result(guide.size());
    while (std::getline(file, word, ' ')) {

        //Modifies word the same way as in training data
        word = remove_debris(word);

        if (guide.count(word) > 0) {
            int index = guide[word];
            result[index] = 1;
        }
    }
    return result;
}