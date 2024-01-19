// project2.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "src/config.h"
#include "src/train_model.h"
#include "src/test_data.h"
#include "src/data_manager.h"
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



//struct to keep metric values tidy (not tidy enough though)
struct Metrics {
    std::string type;
    double precision;
    double recall;
    double f1;
    double accuracy;
    int files;
};

//vectors for testing and training results on each test,
//note that they are still split in negative and positive with metrics.type, 
//but we're relying on order of insertion to make sure they're plotted correctly
std::vector<Metrics> results_test, results_train;

//Metrics calculations, the same calculation is used for both classes, however the arguments given are reversed.
//probably counter intuitive naming 
//(tp : true positiives), (fp : false positives), (tn : true negatives), (fn : false negatives)
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

//Creates a training model with current meta-data
TrainingData create_model(int i, fs::path path_to_file) {
    std::cout << "\nPK = " << PK << " , PN = " << PN << "\nIG top ratio = " << SHED_RATIO << std::endl;
    return train(path_to_file, (i * MODEL_FILES_INCREMENT) + STARTING_FILES);
}

//Basically the program runner, trains, calculates results, saves metric results in vectors
bool train_and_evaluate(fs::path train_dir, fs::path test_dir){

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

//trains more models, with varying k and n for better accuracy (doesn't really work that well)
void perform_full_training(fs::path train_dir, fs::path test_dir) {
    std::cout << "Training commencing.." << std::endl;
    // Start from the given values
    float best_pk = PK; 
    float best_pn = PN; 
    float best_accuracy = 0.0f;

    // Define ranges around the current best values for PK and PN always larger than 0
    float pk_start = std::max(best_pk - (PK_STEP*2), 0.0f); 
    float pk_end = std::max(best_pk + (PK_STEP*2), 0.0f);
    float pn_start = std::max(best_pn - (PN_STEP*2), 0.0f);
    float pn_end = std::max(best_pn + (PN_STEP*2), 0.0f); 

    if (FULL_TRAIN) {
        std::cout << "\nFull training mode." << "\nTraining values: \nStarting PK : " << PK << ", Starting PN : " << PN << "\nTop IG Ratio : " << SHED_RATIO << std::endl;
        //PK values 
        std::cout << "\nOptimizing PK\n" << std::endl;
        for (float pk = pk_start; pk <= pk_end; pk += 0.005f) {
            std::cout << "Current model value : " << PK << std::endl;
            PK = pk;
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

        // PN values using best PK found
        best_accuracy = 0.0f;
        std::cout << "\nOptimizing PN\n" << std::endl;
        for (float pn = pn_start; pn <= pn_end; pn += 0.01f) {
            std::cout << "Current model value : " << PN << std::endl;
            PN = pn;
            TrainingData model = create_model(0, train_dir);
            auto eval_positive = evaluate_dir_reviews(model, test_dir / "pos");
            auto eval_negative = evaluate_dir_reviews(model, test_dir / "neg");
            //doesn't use metrics because we only care about accuracy (maybe correct)
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

//Output function, 
//Prints result metrics in .txt with not an ideal format,
//calls a python script (since I couldn't really make matplotlibcpp work)
//that prints results in plots and csv
void print_results() {
    
    //most of these results are for plotting, which isn't currently implemented, but are kept here for the future
    std::ofstream test_accuracy_txt(OUTPUT_DIR/TEST_TXT_NAME);
    /*
    std::vector<float> test_accuracy_data(MODELS_TO_BE_TRAINED * 2 + 1, 0.0f);
    std::vector<float> test_precision_data(MODELS_TO_BE_TRAINED * 2 + 1, 0.0f);
    std::vector<float> test_recall_data(MODELS_TO_BE_TRAINED * 2 + 1, 0.0f);
    std::vector<float> test_f1_data(MODELS_TO_BE_TRAINED * 2 + 1, 0.0f);
    std::vector<float> test_files(MODELS_TO_BE_TRAINED * 2 + 1, 0.0f);
    */
    int zilly_counter = 1;

    if (test_accuracy_txt.is_open()) {
        test_accuracy_txt << "Test data";
    }

    for (auto& metrics : results_test) {
        /*
        test_files[zilly_counter] = metrics.files;
        test_accuracy_data[zilly_counter] = metrics.accuracy;
        test_precision_data[zilly_counter] = metrics.precision;
        test_recall_data[zilly_counter] = metrics.recall;
        test_f1_data[zilly_counter] = metrics.f1;
        */
        if (test_accuracy_txt.is_open()) {
            test_accuracy_txt << "\n\ntype:test\nnumber:" << zilly_counter << "\nclass:" << metrics.type << std::endl << "\nMetrics:" << std::endl
                << "Precision: " << metrics.precision << std::endl << "Recall: " << metrics.recall << std::endl
                << "F1: " << metrics.f1 << std::endl << "Accuracy: " << metrics.accuracy << std::endl
                << "Files: " << metrics.files << std::endl << "PK = " << PK << ", PN = " << PN << ", IG ratio : " << SHED_RATIO;
        }
        if (metrics.type == "negative") zilly_counter++;
    }

    test_accuracy_txt.close();
    zilly_counter = 1;

    std::ofstream training_accuracy_txt(OUTPUT_DIR / TRAIN_TXT_NAME);
    /*
    std::vector<float> training_accuracy_data(MODELS_TO_BE_TRAINED * 2 + 1, 0.0f);
    std::vector<float> training_precision_data(MODELS_TO_BE_TRAINED * 2 + 1, 0.0f);
    std::vector<float> training_recall_data(MODELS_TO_BE_TRAINED * 2 + 1, 0.0f);
    std::vector<float> training_f1_data(MODELS_TO_BE_TRAINED * 2 + 1, 0.0f);
    std::vector<float> training_files(MODELS_TO_BE_TRAINED * 2 + 1, 0.0f);
    */
    if (training_accuracy_txt.is_open()) {
        training_accuracy_txt << "Training data";
    }

    for (auto& metrics : results_train) {
        /*
        training_files[zilly_counter] = metrics.files;
        training_accuracy_data[zilly_counter] = metrics.accuracy;
        training_precision_data[zilly_counter] = metrics.precision;
        training_recall_data[zilly_counter] = metrics.recall;
        training_f1_data[zilly_counter] = metrics.f1;
        */
        if (training_accuracy_txt.is_open()) {
            training_accuracy_txt << "\n\ntype:train\nnumber:" << zilly_counter << "\nclass:" << metrics.type << std::endl << "\nMetrics:" << std::endl
                << "Precision: " << metrics.precision << std::endl << "Recall: " << metrics.recall << std::endl
                << "F1: " << metrics.f1 << std::endl << "Accuracy: " << metrics.accuracy << std::endl
                << "Files: " << metrics.files << std::endl << "PK = " << PK << ", PN = " << PN << ", IG ratio : " << SHED_RATIO;
        }
        if (metrics.type == "negative") zilly_counter++;

    }
    training_accuracy_txt.close();

    std::string py_script_run_str = "python " + PY_SCRIPT_DIR.string() + " " + 
        (FROM_SCRIPT_TO / TEST_TXT_NAME).string() + " " + 
        (FROM_SCRIPT_TO / TRAIN_TXT_NAME).string();

    const char* py_script_run = py_script_run_str.c_str();
    system(py_script_run);
    /*
    plt::title("Learning curve of test and train files");
    plt::plot(training_files, training_accuracy_data, "Train data");
    plt::plot(test_files, test_accuracy_data, "Train data");
    plt::save("./data/test-train_graph.png");
    */
}

void check_variables() {
    if (LOAD_MODELS) return;

    if (STARTING_FILES <= 0.0f) {
        std::cout << "The STARTING_FILES variable should be greater than 0.\n";
        return;
    }
    if (MODEL_FILES_INCREMENT < 0) {
        std::cout << "The MODEL_FILES_INCREMENT variable is not greater than 0, defaulting to 1 model training.\n";
        MODELS_TO_BE_TRAINED = 1;
        MODEL_FILES_INCREMENT = 0;
    }
    if (MODELS_TO_BE_TRAINED < 0) {
        std::cout << "The MODELS_TO_BE_TRAINED variable is less than 0, defaulting to 1.\n";
        MODELS_TO_BE_TRAINED = 1;
    }
}

int main(int argc, char* argv[]) {

    //check config.h variables given normalizes some values if wrong input is given, 
    // doesn't check everything, program could still break if variables aren't correct
    check_variables();

    //loads stopwords to ignore during word selection
    //default list is from https://gist.github.com/sebleier/554280 (NLTK)
    load_stopwords();

    //if executed with an argument it is taken as the path to the dataset
    fs::path train_dir, test_dir;
    if (argc > 1) {
        fs::path original_path(argv[1]);
        train_dir = original_path / DATASET_DIR / "train";
        test_dir = original_path / DATASET_DIR / "test";
    }
    else {
        train_dir = DATASET_DIR / "train";
        test_dir = DATASET_DIR / "test";
    }

    //checks if varying K and N values should be tested before training the model
    if (FULL_TRAIN) {
        perform_full_training(train_dir, test_dir);
    }
    else {
        train_and_evaluate(train_dir, test_dir);
    }
    
    //prints results: plots in png and tables in .txt
    print_results();


    return 0;
}


