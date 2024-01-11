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

struct Metrics {
    std::string type;
    double precision;
    double recall;
    double f1;
    double accuracy;
    int files;
};

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
bool FULL_TRAIN;


std::vector<Metrics> results_test, results_train;

void read_config(const std::string& filename = "config.txt"){

    std::ifstream config_file(filename);
    if (!config_file.is_open()) {
        std::cerr << "Unable to open config file: " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(config_file, line)) {
        // Ignore comments and empty lines
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string key, value;

        if (std::getline(iss, key, '=') && std::getline(iss, value)) {
            if (key == "FULL_TRAIN") FULL_TRAIN = (value == "true" || value == "1");
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

Metrics calculate_metrics(int tp, int fp, int tn, int fn, std::string type) {
    Metrics metrics;

    metrics.type = type;
    metrics.files = (tp + fp + tn + fn);
    metrics.precision = tp / (double)(tp + fp);
    metrics.recall = tp / (double)(tp + fn);
    metrics.f1 = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall);
    metrics.accuracy = (double)(tp + tn) / (double)(metrics.files);
    
    return metrics;
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

bool train_and_evaluate(fs::path train_dir, fs::path test_dir){
    //Checking arguments and creating paths for readability if no arguments are given, reader is run on current directory.


    fs::path negative_test = (test_dir / "neg");
    fs::path positive_test = (test_dir / "pos");

    Training_data* model = nullptr;

    std::pair<int, int> evaluation_results_positive, evaluation_results_train_positive;
    std::pair<int, int> evaluation_results_negative, evaluation_results_train_negative;
    int training_files;
    
    for (int i = 0; i <= MODELS_TO_BE_TRAINED - 1; ++i) {

        model = create_model(i, train_dir);

        training_files = model->positive_file_count + model->negative_file_count;
        evaluation_results_positive = evaluate_dir_reviews(positive_test, model);
        evaluation_results_negative = evaluate_dir_reviews(negative_test, model);
        evaluation_results_train_positive = evaluate_dir_reviews(train_dir / "pos", model);
        evaluation_results_train_negative = evaluate_dir_reviews(train_dir / "neg", model);
        Metrics positive_metrics = calculate_metrics(evaluation_results_positive.first, evaluation_results_positive.second, evaluation_results_negative.second, evaluation_results_negative.first, "positive");
        Metrics negative_metrics = calculate_metrics(evaluation_results_negative.second, evaluation_results_negative.first, evaluation_results_positive.first, evaluation_results_positive.second, "negative");

        Metrics positive_train_metrics = calculate_metrics(evaluation_results_train_positive.first, evaluation_results_train_positive.second, evaluation_results_train_negative.second, evaluation_results_train_negative.first, "positive ");
        Metrics negative_train_metrics = calculate_metrics(evaluation_results_train_negative.second, evaluation_results_train_negative.first, evaluation_results_train_positive.first, evaluation_results_train_positive.second, "negative");

        results_test.push_back(positive_metrics);
        results_test.push_back(negative_metrics);
        results_train.push_back(positive_train_metrics);
        results_train.push_back(negative_train_metrics);
    }
    model = nullptr;
    return 1;
}

void perform_full_training(fs::path train_dir, fs::path test_dir) {
    // Start from the current best known values
    float best_pk = PK; // e.g., 0.05
    float best_pn = PN; // e.g., 2
    float best_accuracy = 0.0f;

    // Define ranges around the current best values for PK and PN
    float pk_start = best_pk - 0.02f; // e.g., start a bit lower than the current best
    float pk_end = best_pk + 0.02f; // e.g., end a bit higher than the current best
    float pn_start = best_pn - 0.4f; // e.g., start a bit lower than the current best
    float pn_end = best_pn + 0.4f; // e.g., end a bit higher than the current best

    if (FULL_TRAIN) {
        // Loop over PK values
        for (float pk = pk_start; pk <= pk_end; pk += 0.01f) {
            PK = pk; // Set PK
            Training_data* model = create_model(0, train_dir);
            auto eval_positive = evaluate_dir_reviews(test_dir / "pos", model);
            auto eval_negative = evaluate_dir_reviews(test_dir / "neg", model);
            int correct_predictions = eval_positive.first + eval_negative.second;
            int total_predictions = eval_positive.first + eval_positive.second + eval_negative.first + eval_negative.second;
            float accuracy = static_cast<float>(correct_predictions) / static_cast<float>(total_predictions);

            if (accuracy > best_accuracy) {
                best_accuracy = accuracy;
                best_pk = pk;
            }
            delete model;
        }

        // Loop over PN values using the best PK found
        best_accuracy = 0.0f;
        for (float pn = pn_start; pn <= pn_end; pn += 0.2f) {
            PN = pn;
            Training_data* model = create_model(0, train_dir);
            auto eval_positive = evaluate_dir_reviews(test_dir / "pos", model);
            auto eval_negative = evaluate_dir_reviews(test_dir / "neg", model);
            int correct_predictions = eval_positive.first + eval_negative.second;
            int total_predictions = eval_positive.first + eval_positive.second + eval_negative.first + eval_negative.second;
            float accuracy = static_cast<float>(correct_predictions) / static_cast<float>(total_predictions);

            if (accuracy > best_accuracy) {
                best_accuracy = accuracy;
                best_pn = pn;
            }
            delete model;
        }

        PK = best_pk;
        PN = best_pn;

        std::cout << "Best PK: " << PK << ", Best PN: " << PN << ", with accuracy: " << best_accuracy << std::endl;

        // Train models with the best PK and PN and increasing amounts of data
        for (int i = 0; i < 10; ++i) {
            float train_amount = 0.2f + i * 0.05f;
            STARTING_FILES = train_amount;
            PK = best_pk;
            PN = best_pn;
            train_and_evaluate(train_dir, test_dir);
        }
    }
}


void print_results() {
    // Plotting graph
    std::ofstream test_accuracy_txt("data/test_data_accuracy.txt");
    int zilly_counter = 0;
    if (test_accuracy_txt.is_open()) {
        test_accuracy_txt << "Test data" << std::endl;
    }


    for (auto& metrics : results_test) {
        if (test_accuracy_txt.is_open()) {
            test_accuracy_txt << "Test " << zilly_counter++ << metrics.type << std::endl << " Metrics:" << std::endl
                  << "Precision: " << metrics.precision << std::endl << ", Recall: " << metrics.recall << std::endl
                  << ", F1: " << metrics.f1 << std::endl << ", Accuracy: " << metrics.accuracy << std::endl
                  << ", Files: " << metrics.files << std::endl;
        }
    }

    test_accuracy_txt.close();
    zilly_counter = 0;
    //plt::title("Learning curve of test and train files");
    //plt::named_plot("Test dir", amount_of_files, model_accuracy);
    //plt::plot(amount_of_files, model_accuracy, "Test data");

    std::ofstream training_accuracy_txt("data/training_data_accuracy.txt");
    if (training_accuracy_txt.is_open()) {
        training_accuracy_txt << "Training data" << std::endl;
    }

    for (auto& metrics : results_train) {
        if (training_accuracy_txt.is_open()) {
            training_accuracy_txt << "Training " << zilly_counter++ << metrics.type << std::endl << " Metrics:" << std::endl
                << "Precision: " << metrics.precision << std::endl << ", Recall: " << metrics.recall << std::endl
                << ", F1: " << metrics.f1 << std::endl << ", Accuracy: " << metrics.accuracy << std::endl
                << ", Files: " << metrics.files << std::endl;
        }
    }
    training_accuracy_txt.close();

    //plt::legend();
    //plt::named_plot("Train dir", amount_of_files, model_accuracy);
    //plt::plot(amount_of_files, model_accuracy, "Train data");
    //plt::save("./data/test-train_graph.png");
}

int main(int argc, char* argv[]) {
    read_config("config.txt");
    if (!check_variables()) {
        std::cerr << "Variable check failed. Exiting." << std::endl;
        return -1;
    }

    fs::path train_dir, test_dir;
    if (argc > 1) {
        fs::path original_path(argv[1]);
        train_dir = original_path / MAIN_DIR / "train";
        test_dir = original_path / MAIN_DIR / "test";
    }
    else {
        train_dir = MAIN_DIR / "train";
        test_dir = MAIN_DIR / "test";
    }

    if (FULL_TRAIN) {
        perform_full_training(train_dir, test_dir);
    }
    else {
        train_and_evaluate(train_dir, test_dir);
    }

    print_results();
    return 0;
}



    
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
