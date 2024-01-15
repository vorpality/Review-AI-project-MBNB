// project2.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "config.h"
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
#include <algorithm>
#include <cstdlib>
//#include "vcpkg.matplotlibcpp.h" 

namespace fs = std::filesystem;
//namespace plt = matplotlibcpp;

const std::filesystem::path MAIN_DIR = "x64/Debug/includes/aclImdb";

struct Metrics {
    std::string type;
    double precision;
    double recall;
    double f1;
    double accuracy;
    int files;
};



std::vector<Metrics> results_test, results_train;


Metrics calculate_metrics(int tp, int fp, int tn, int fn, std::string type, int files) {
    Metrics metrics;

    metrics.type = type;
    metrics.files = files;
    metrics.precision = tp / (double)(tp + fp);
    metrics.recall = tp / (double)(tp + fn);
    metrics.f1 = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall);
    metrics.accuracy = (double)(tp) / (double)(tp + fp);
    
    return metrics;
}


TrainingData create_model(int i, fs::path path_to_file) {
    std::cout << "\nPK = " << PK << " , PN = " << PN << "\nIG top ratio = " << SHED_RATIO << std::endl;
    return train(path_to_file, (i * MODEL_FILES_INCREMENT) + STARTING_FILES);
}

bool train_and_evaluate(fs::path train_dir, fs::path test_dir){
    //Checking arguments and creating paths for readability if no arguments are given, reader is run on current directory.


    fs::path negative_test = (test_dir / "neg");
    fs::path positive_test = (test_dir / "pos");


    std::pair<int, int> evaluation_results_positive, evaluation_results_train_positive;
    std::pair<int, int> evaluation_results_negative, evaluation_results_train_negative;
    int training_files;
    
    for (int i = 0; i <= MODELS_TO_BE_TRAINED - 1; ++i) {
        std::cout << "\nTraining model " << i + 1 << " out of " << MODELS_TO_BE_TRAINED << std::endl;
            
        TrainingData model = create_model(i, train_dir);

        training_files = model.get_total_files();
        evaluation_results_positive = evaluate_dir_reviews(model, positive_test);
        evaluation_results_negative = evaluate_dir_reviews(model, negative_test);
        evaluation_results_train_positive = evaluate_dir_reviews(model, "", "pos");
        evaluation_results_train_negative = evaluate_dir_reviews(model, "", "neg");
        Metrics positive_metrics = calculate_metrics(evaluation_results_positive.first, evaluation_results_positive.second, evaluation_results_negative.second, evaluation_results_negative.first, "positive", training_files);
        Metrics negative_metrics = calculate_metrics(evaluation_results_negative.second, evaluation_results_negative.first, evaluation_results_positive.first, evaluation_results_positive.second, "negative", training_files);

        Metrics positive_train_metrics = calculate_metrics(evaluation_results_train_positive.first, evaluation_results_train_positive.second, evaluation_results_train_negative.second, evaluation_results_train_negative.first, "positive", training_files);
        Metrics negative_train_metrics = calculate_metrics(evaluation_results_train_negative.second, evaluation_results_train_negative.first, evaluation_results_train_positive.first, evaluation_results_train_positive.second, "negative", training_files);
        std::cout << "\nTraining files used : " << training_files << "\nResults on Test files :\n" <<"Total Accuracy : " << (positive_metrics.accuracy + negative_metrics.accuracy)/2 << " , Positive accuracy : " << positive_metrics.accuracy << " , Negative accuracy : " << negative_metrics.accuracy << std::endl;
        std::cout << "\nResults on Training files :\n" << "Total Accuracy : " << (positive_train_metrics.accuracy + negative_train_metrics.accuracy)/2 << " , Positive accuracy : " << positive_train_metrics.accuracy << " , Negative accuracy : " << negative_train_metrics.accuracy << std::endl;

        results_test.push_back(positive_metrics);
        results_test.push_back(negative_metrics);
        results_train.push_back(positive_train_metrics);
        results_train.push_back(negative_train_metrics);
    }
    return 1;
}

void perform_full_training(fs::path train_dir, fs::path test_dir) {
    std::cout << "Training commencing.." << std::endl;
    // Start from the current best known values
    float best_pk = PK; // e.g., 0.05
    float best_pn = PN; // e.g., 2
    float best_accuracy = 0.0f;

    // Define ranges around the current best values for PK and PN always larger than 0
    float pk_start = std::max(best_pk - (PK_STEP*2), 0.0f); //  start a bit lower than the current best
    float pk_end = std::max(best_pk + (PK_STEP*2), 0.0f); // end a bit higher than current pk
    float pn_start = std::max(best_pn - (PN_STEP*2), 0.0f); // start a bit lower than the current best
    float pn_end = std::max(best_pn + (PN_STEP*2), 0.0f); // end a bit higher than the current best

    if (FULL_TRAIN) {
        std::cout << "\nFull training mode." << "\nTraining values: \nStarting PK : " << PK << ", Starting PN : " << PN << "\nTop IG Ratio : " << SHED_RATIO << std::endl;
        // Loop over PK values  q1
        std::cout << "\nOptimizing PK\n" << std::endl;
        for (float pk = pk_start; pk <= pk_end; pk += 0.005f) {
            std::cout << "Current model value : " << PK << std::endl;
            PK = pk; // Set PK
            TrainingData model = create_model(0, train_dir);
            auto eval_positive = evaluate_dir_reviews(model, test_dir / "pos");
            auto eval_negative = evaluate_dir_reviews(model, test_dir / "neg");
            int correct_predictions = eval_positive.first + eval_negative.second;
            int total_predictions = eval_positive.first + eval_positive.second + eval_negative.first + eval_negative.second;
            float accuracy = static_cast<float>(correct_predictions) / static_cast<float>(total_predictions);
            std::cout << "Accuracy with PK = " << PK << " : " << accuracy << std::endl;
            if (accuracy > best_accuracy) {
                best_accuracy = accuracy;
                best_pk = pk;
            }
        }
        PK = best_pk;
        std::cout << "\nBest accuracy found with PK = " << PK << " : " << best_accuracy <<std::endl;

        // Loop over PN values using the best PK found
        best_accuracy = 0.0f;
        std::cout << "\nOptimizing PN\n" << std::endl;
        for (float pn = pn_start; pn <= pn_end; pn += 0.01f) {
            std::cout << "Current model value : " << PN << std::endl;
            PN = pn;
            TrainingData model = create_model(0, train_dir);
            auto eval_positive = evaluate_dir_reviews(model, test_dir / "pos");
            auto eval_negative = evaluate_dir_reviews(model, test_dir / "neg");
            int correct_predictions = eval_positive.first + eval_negative.second;
            int total_predictions = eval_positive.first + eval_positive.second + eval_negative.first + eval_negative.second;
            float accuracy = static_cast<float>(correct_predictions) / static_cast<float>(total_predictions);
            std::cout << "Accuracy with PN = " << PN << " : " << accuracy << std::endl;

            if (accuracy > best_accuracy) {
                best_accuracy = accuracy;
                best_pn = pn;
            }
        }
        std::cout << "\nBest accuracy found with PN = " << PN << " : " << best_accuracy <<std::endl;

        PN = best_pn;

        std::cout << "\nBest PK: " << PK << ", Best PN: " << PN << ", with accuracy: " << best_accuracy << std::endl;


        train_and_evaluate(train_dir, test_dir);
        
    }
}


void print_results() {
    // Plotting graph
    std::ofstream test_accuracy_txt("data/test_data_accuracy.txt");
    std::vector<float> test_accuracy_data(MODELS_TO_BE_TRAINED * 2 + 1, 0.0f);
    std::vector<float> test_precision_data(MODELS_TO_BE_TRAINED * 2 + 1, 0.0f);
    std::vector<float> test_recall_data(MODELS_TO_BE_TRAINED * 2 + 1, 0.0f);
    std::vector<float> test_f1_data(MODELS_TO_BE_TRAINED * 2 + 1, 0.0f);
    std::vector<float> test_files(MODELS_TO_BE_TRAINED * 2 + 1, 0.0f);
    int zilly_counter = 0;

    if (test_accuracy_txt.is_open()) {
        test_accuracy_txt << "Test data";
    }

    for (auto& metrics : results_test) {
        test_files[zilly_counter] = metrics.files;
        test_accuracy_data[zilly_counter] = metrics.accuracy;
        test_precision_data[zilly_counter] = metrics.precision;
        test_recall_data[zilly_counter] = metrics.recall;
        test_f1_data[zilly_counter] = metrics.f1;
        if (test_accuracy_txt.is_open()) {
            test_accuracy_txt << "\n\nTest " << zilly_counter++ << " " << metrics.type << std::endl << "\nMetrics:" << std::endl
                << "Precision: " << metrics.precision << std::endl << "Recall: " << metrics.recall << std::endl
                << "F1: " << metrics.f1 << std::endl << "Accuracy: " << metrics.accuracy << std::endl
                << "Files: " << metrics.files << std::endl << "PK = " << PK << ", PN = " << PN << ", IG ratio : " << SHED_RATIO;
        }
    }
    system("python data/plotting.py");
    test_accuracy_txt.close();
    zilly_counter = 0;

    std::ofstream training_accuracy_txt("data/training_data_accuracy.txt");
    std::vector<float> training_accuracy_data(MODELS_TO_BE_TRAINED * 2 + 1, 0.0f);
    std::vector<float> training_precision_data(MODELS_TO_BE_TRAINED * 2 + 1, 0.0f);
    std::vector<float> training_recall_data(MODELS_TO_BE_TRAINED * 2 + 1, 0.0f);
    std::vector<float> training_f1_data(MODELS_TO_BE_TRAINED * 2 + 1, 0.0f);
    std::vector<float> training_files(MODELS_TO_BE_TRAINED * 2 + 1, 0.0f);
    if (training_accuracy_txt.is_open()) {
        training_accuracy_txt << "Training data";
    }

    for (auto& metrics : results_train) {
        training_files[zilly_counter] = metrics.files;
        training_accuracy_data[zilly_counter] = metrics.accuracy;
        training_precision_data[zilly_counter] = metrics.precision;
        training_recall_data[zilly_counter] = metrics.recall;
        training_f1_data[zilly_counter] = metrics.f1;

        if (training_accuracy_txt.is_open()) {
            training_accuracy_txt << "\n\nTraining " << zilly_counter++ << " " << metrics.type << std::endl << "\nMetrics:" << std::endl
                << "Precision: " << metrics.precision << std::endl << "Recall: " << metrics.recall << std::endl
                << "F1: " << metrics.f1 << std::endl << "Accuracy: " << metrics.accuracy << std::endl
                << "Files: " << metrics.files << std::endl << "PK = " << PK << ", PN = " << PN << ", IG ratio : " << SHED_RATIO;
        }
    }
    training_accuracy_txt.close();

    /*
    plt::title("Learning curve of test and train files");
    plt::plot(training_files, training_accuracy_data, "Train data");
    plt::plot(test_files, test_accuracy_data, "Train data");
    plt::save("./data/test-train_graph.png");
    */
}

int main(int argc, char* argv[]) {
    check_variables();
    load_stopwords();
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
    system("pause");
    return 0;
}


