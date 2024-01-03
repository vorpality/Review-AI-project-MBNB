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
    fs::path test_dir = (original_path / MAIN_DIR / "test" /"pos");

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
    int positives = 0;
    int negatives = 0;
    //BernouliNB evaluation

    float Pc1 = (float)model->positive_file_count;
    float Pc0 = (float)model->negative_file_count;

    for (auto const& dir_entry : std::filesystem::directory_iterator(test_dir)) {
        review = file_to_vector(dir_entry, model->word_index_guide);
        float result = test_vector(review, model);

        //std::cout << "Ppos: " << Ppos << " Pneg: " << Pneg << std::endl;
        if (result > 0) {
            positives++;
        }
        else {
            negatives++;
        }
        std::string predicted = (result > 0) ? "Positive" : "Negative";
        std::cout <<  dir_entry << " Predicted : " << predicted << std::endl;
        if (output.is_open()) {
            output << dir_entry << " Predicted : " << predicted << std::endl;
        }
    }
    std::cout << "Positives : " << positives <<" Negatives : " << negatives << std::endl;
    output.close();
    delete model;
    return 1;
   
}

