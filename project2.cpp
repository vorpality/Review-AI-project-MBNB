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
#include <matplotlibcpp.h>

namespace fs = std::filesystem;
namespace plt = matplotlibcpp;

const std::filesystem::path MAIN_DIR = "x64/Debug/includes/aclImdb";

/* 
    starting_files = the percentage of files the first model will be trained on(the least amount of training files to be taken into consideration)
    models_to_be_trained = the amount of models that will be trained.
    model_files_increment = the value by which the amount of training files will be incremented in every model.
    save_models = true saves the models for future use
    load_models = true loads the models if they exist, this makes the execution ignore all other control variables given 

    example : 
    starting file = 0.04
    models_to_be_trained = 10
    model_files_increment = 0.1f

    10 models will be trained.
    the first model will be trained on 4% of the given data
    the second model will be trained on 14% of the given data
    the third model will be trained on 24% of the given data, etc
*/

float starting_files = 0.04;
int models_to_be_trained = 10;
float model_files_increment = 0.05f;
bool save_models = false;
bool load_models = false;
fs::path load_dir = "data/model";

std::vector<std::pair<double, int>> results_test, results_train;

bool check_variables() {
    if (load_models) return 1;

    if (starting_files <= 0.0) {
        std::cout << "The starting_files variable given should be greater than 0";
        return 0;
    }
    if (starting_files > 1) {
        std::cout << "The starting_files variable given is greater than 1, so it has been defaulted to 1";
        starting_files = 1;
        return 1;
    }
    if (model_files_increment < 0) {
        std::cout << "The model_files_increment variable given is not greater than 0, so it has been defaulted to 0.1";
        model_files_increment = 0.1;
        return 1;
    }
    if (models_to_be_trained < 0) {
        std::cout << "The models_to_be_trained variable given is less than 0, so it has been defaulted to 1";
        models_to_be_trained = 1;
        return 1;
    }
    if (starting_files + (models_to_be_trained * model_files_increment) > 1) {
        std::cout << "The program cannot continue with the set of variables given, check \"starting_files\", \"models_to_be_trained\" and \"model_files_increment\".";
        return 0;
    }
    return 1;
}

Training_data* create_model(int i, fs::path path_to_file) {
    if (load_models) {
        std::string file_name = load_dir.string() + std::to_string(i);
        return load_training_data(file_name);
    }
    else {
        Training_data* model = train(path_to_file, (i * model_files_increment) + starting_files);
        if (save_models) {
            std::string file_name = load_dir.string() + std::to_string(i);
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
    
    for (int i = 0; i <= models_to_be_trained - 1; ++i) {

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
int main(int argc, char *argv[])
{
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

