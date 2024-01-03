// project2.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "train_model.h"
#include "test_data.h"
#include "data_manager.h"
#include <iostream>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <filesystem>


const std::filesystem::path MAIN_DIR = "x64/Debug/includes/aclImdb";

namespace fs = std::filesystem;

int main(int argc, char *argv[])
{

    std::string training_data_file = "training_data_checkpoint.bin";
    //Checking arguments and creating paths for readability if no arguments are given, reader is run on current directory.
    if (argc > 3) {
        return -1;
    }

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
    }

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
   
}

