// project2.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "train_model.h"
#include "test_data.h"
#include "data_manager.h"
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <filesystem>
#include <matplotlibcpp.h>

namespace fs = std::filesystem;
namespace plt = matplotlibcpp;

const std::filesystem::path MAIN_DIR = "x64/Debug/includes/aclImdb";

//Globals

float PK;
float PN;
float STARTING_FILES;
int MODELS_TO_BE_TRAINED;
float MODEL_FILES_INCREMENT;
bool SAVE_MODELS;
bool LOAD_MODELS;
std::filesystem::path LOAD_DIR;
int FILE_CAP;
int MINIMUM_LETTERS;


void readConfig(const std::string& filename = "config.txt"){

    std::ifstream configFile(filename);
    if (!configFile.is_open()) {
        std::cerr << "Unable to open config file: " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(configFile, line)) {
        // Ignore comments and empty lines
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string key, value;

        if (std::getline(iss, key, '=') && std::getline(iss, value)) {
            if (key == "PK") PK = std::stof(value);
            else if (key == "PN") PN = std::stof(value);
            else if (key == "STARTING_FILES") STARTING_FILES = std::stof(value);
            else if (key == "MODELS_TO_BE_TRAINED") MODELS_TO_BE_TRAINED = std::stoi(value);
            else if (key == "MODEL_FILES_INCREMENT") MODEL_FILES_INCREMENT = std::stof(value);
            else if (key == "SAVE_MODELS") SAVE_MODELS = (value == "true" || value == "1");
            else if (key == "LOAD_MODELS") LOAD_MODELS = (value == "true" || value == "1");
            else if (key == "LOAD_DIR") LOAD_DIR = value;
            else if (key == "FILE_CAP") FILE_CAP = std::stoi(value);
            else if (key == "MINIMUM_LETTERS") MINIMUM_LETTERS = std::stoi(value);
        }
    }
}

std::vector<std::pair<double, int>> results_test, results_train;

bool check_variables() {
    if (LOAD_MODELS) return true;

    if (STARTING_FILES <= 0.0) {
        std::cout << "The STARTING_FILES variable given in the config.txt file should be greater than 0.\n";
        return false;
    }
    if (STARTING_FILES > 1) {
        std::cout << "The STARTING_FILES variable given in the config.txt file is greater than 1, so it has been defaulted to 1.\n";
        STARTING_FILES = 1;
    }
    if (MODEL_FILES_INCREMENT < 0) {
        std::cout << "The MODEL_FILES_INCREMENT variable given in the config.txt file is not greater than 0, so it has been defaulted to 0.1.\n";
        MODEL_FILES_INCREMENT = 0.1;
    }
    if (MODELS_TO_BE_TRAINED < 0) {
        std::cout << "The MODELS_TO_BE_TRAINED variable given in the config.txt file is less than 0, so it has been defaulted to 1.\n";
        MODELS_TO_BE_TRAINED = 1;
    }
    if (STARTING_FILES + (MODELS_TO_BE_TRAINED * MODEL_FILES_INCREMENT) > 1) {
        std::cout << "The program cannot continue with the set of variables given in the config.txt file, check \"STARTING_FILES\", \"MODELS_TO_BE_TRAINED\" and \"MODEL_FILES_INCREMENT\".\n";
        return false;
    }
    return true;
}

Training_data* create_model(int i, fs::path path_to_file) {
    if (LOAD_MODELS) {
        std::string file_name = LOAD_DIR.string() + std::to_string(i);
        return load_training_data(file_name);
    }
    else {
        Training_data* model = train(path_to_file, (i * MODEL_FILES_INCREMENT) + STARTING_FILES);
        if (SAVE_MODELS) {
            std::string file_name = LOAD_DIR.string() + std::to_string(i);
            save_training_data(model, file_name);
        }
        return model;
    }

}

bool train_and_evaluate(int argument_count, char* argument_values [] ) {
    //Checking arguments and creating paths for readability if no arguments are given, reader is run on current directory.
    if (argument_count > 3) {
        return 0;
    }

    fs::path original_path = (argument_count == 2) ? fs::path(argument_values[1]) : fs::path("");
    fs::path train_dir = original_path / MAIN_DIR / "train";
    fs::path test_dir = (original_path / MAIN_DIR / "test");
    fs::path negative_test = (test_dir / "neg");
    fs::path positive_test = (test_dir / "pos");

    Training_data* model = nullptr;

    std::pair<int, int> evaluation_results_positive, evaluation_results_train_positive;
    std::pair<int, int> evaluation_results_negative, evaluation_results_train_negative;
    int correct_results_positive;
    int correct_results_negative;
    int total;
    float accuracy;
    int training_files;
    
    for (int i = 0; i <= MODELS_TO_BE_TRAINED - 1; ++i) {

        model = create_model(i, train_dir);

        training_files = model->positive_file_count + model->negative_file_count;
        evaluation_results_positive = evaluate_dir_reviews(positive_test, model);
        evaluation_results_negative = evaluate_dir_reviews(negative_test, model);
        evaluation_results_train_positive = evaluate_dir_reviews(train_dir / "pos", model);
        evaluation_results_train_negative = evaluate_dir_reviews(train_dir / "neg", model);
        correct_results_positive = evaluation_results_positive.first;
        correct_results_negative = evaluation_results_negative.second;
        total = correct_results_positive + correct_results_negative + evaluation_results_positive.second + evaluation_results_negative.first;
        accuracy = (double)(correct_results_positive + correct_results_negative) / (double)total;
        results_test.push_back(std::pair <double, int>(accuracy, training_files));

        correct_results_positive = evaluation_results_train_positive.first;
        correct_results_negative = evaluation_results_train_negative.second;
        total = correct_results_positive + correct_results_negative + evaluation_results_train_positive.second + evaluation_results_train_negative.first;
        accuracy = (double)(correct_results_positive + correct_results_negative) / (double)total;
        results_train.push_back(std::pair <double, int>(accuracy, training_files));
    }
    model = nullptr;
    return 1;
}

void print_results() {
    // Plotting graph
    std::vector<float> model_accuracy;
    std::vector<float> amount_of_files;

    std::ofstream test_accuracy_txt("data/test_data_accuracy.txt");
    if (test_accuracy_txt.is_open()) {
        test_accuracy_txt << "Test data" << std::endl;
    }


    for (auto& pair : results_test) {
        if (test_accuracy_txt.is_open()) {
            test_accuracy_txt << "Training files : " << pair.second << " , Accuracy : " << pair.first << std::endl;
        }

        model_accuracy.push_back(pair.first);
        amount_of_files.push_back(pair.second);
    }

    test_accuracy_txt.close();
    
    //plt::title("Learning curve of test and train files");
    //plt::named_plot("Test dir", amount_of_files, model_accuracy);
    //plt::plot(amount_of_files, model_accuracy, "Test data");
    model_accuracy.clear();
    amount_of_files.clear();

    std::ofstream training_accuracy_txt("data/training_data_accuracy.txt");
    if (training_accuracy_txt.is_open()) {
        training_accuracy_txt << "Training data" << std::endl;
    }

    for (auto& pair : results_train) {
        if (training_accuracy_txt.is_open()) {
            training_accuracy_txt << "Training files : " << pair.second << " , Accuracy : " << pair.first << std::endl;
        }

        model_accuracy.push_back(pair.first);
        amount_of_files.push_back(pair.second);
    }
    training_accuracy_txt.close();

    //plt::legend();
    //plt::named_plot("Train dir", amount_of_files, model_accuracy);
    //plt::plot(amount_of_files, model_accuracy, "Train data");
    //plt::save("./data/test-train_graph.png");
}
int main(int argc, char* argv[]) {

    readConfig("config.txt");
    //Check model training variables
    if (!check_variables()) {
        return -1;
    }
    
    if (!train_and_evaluate(argc, argv)) {
        return -1;
    }

    print_results();

    
    //plot_learning_curve(results);
    /*
    std::string training_data_file = "training_data_checkpoint.bin";

    fs::path original_path = (argc == 2) ? fs::path(argv[1]) : fs::path("");
    fs::path train_dir = original_path / MAIN_DIR / "train";


    Training_data* model = nullptr;
    if (is_training_data_available(training_data_file)){
        std::cout << "Training data available. \nLoading.." << std::endl;
        model = load_training_data(training_data_file);
    }
    else {
        std::cout << "Training data not available. \nTraining new model.." << std::endl;
        model = train(train_dir);
        std::cout << "Model has been trained. \nSaving training data to binary file.." << std::endl;
        save_training_data(model, training_data_file);
        std::cout << "Model has been saved." << std::endl;
    }


    std::cout << "Testing dataset.." << std::endl;

    std::vector<int> review;
    std::ofstream output("op.txt");

    fs::path test_dir = (original_path / MAIN_DIR / "test");
    fs::path negative_test = (test_dir / "neg");
    fs::path positive_test = (test_dir / "pos");

    std::pair<int, int> evaluation_results;
    
    evaluation_results = evaluate_dir_reviews(positive_test, model);

    std::string results_string_positive =
        "Positive directory results.\nPositives: " +
        std::to_string(evaluation_results.first) +
        "\nNegatives: " + std::to_string(evaluation_results.second);
    std::cout << results_string_positive << std::endl;

    if (output.is_open()) {
        output << results_string_positive << std::endl;
    }

    evaluation_results = evaluate_dir_reviews(negative_test, model);

    std::string results_string_negative =
        "Negative directory results.\nPositives: " +
        std::to_string(evaluation_results.first) +
        "\nNegatives: " + std::to_string(evaluation_results.second);
    std::cout << results_string_negative << std::endl;

    if (output.is_open()) {
        output << results_string_negative << std::endl;
    }

    output.close();
    delete model;
    return 1;
   */
}

