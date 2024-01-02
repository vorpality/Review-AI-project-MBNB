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



/*void test(fs::path dir, Train_results* training_data) {

    fs::path negative = dir / "neg";
    fs::path positive = dir / "pos";
    for (auto const& dir_entry : std::filesystem::directory_iterator(negative)) {
        tokenize(negative, neg_map, false);
    }
}*/

std::vector<std::pair<size_t, size_t>> get_edges() {
    return {
        {0, 0},   {1, 0},   {5, 0},   {0, 1},   {1, 1},   {2, 1},   {6, 1},
        {1, 2},   {2, 2},   {3, 2},   {7, 2},   {2, 3},   {3, 3},   {4, 3},
        {8, 3},   {3, 4},   {4, 4},   {9, 4},   {0, 5},   {5, 5},   {6, 5},
        {10, 5},  {1, 6},   {5, 6},   {6, 6},   {7, 6},   {11, 6},  {2, 7},
        {6, 7},   {7, 7},   {8, 7},   {12, 7},  {3, 8},   {7, 8},   {8, 8},
        {9, 8},   {13, 8},  {4, 9},   {8, 9},   {9, 9},   {14, 9},  {5, 10},
        {10, 10}, {11, 10}, {15, 10}, {6, 11},  {10, 11}, {11, 11}, {12, 11},
        {16, 11}, {7, 12},  {11, 12}, {12, 12}, {13, 12}, {17, 12}, {8, 13},
        {12, 13}, {13, 13}, {14, 13}, {18, 13}, {9, 14},  {13, 14}, {14, 14},
        {19, 14}, {10, 15}, {15, 15}, {16, 15}, {20, 15}, {11, 16}, {15, 16},
        {16, 16}, {17, 16}, {21, 16}, {12, 17}, {16, 17}, {17, 17}, {18, 17},
        {22, 17}, {13, 18}, {17, 18}, {18, 18}, {19, 18}, {23, 18}, {14, 19},
        {18, 19}, {19, 19}, {24, 19}, {15, 20}, {20, 20}, {21, 20}, {25, 20},
        {16, 21}, {20, 21}, {21, 21}, {22, 21}, {26, 21}, {17, 22}, {21, 22},
        {22, 22}, {23, 22}, {27, 22}, {18, 23}, {22, 23}, {23, 23}, {24, 23},
        {28, 23}, {19, 24}, {23, 24}, {24, 24}, {29, 24}, {20, 25}, {25, 25},
        {26, 25}, {35, 25}, {21, 26}, {25, 26}, {26, 26}, {27, 26}, {36, 26},
        {22, 27}, {26, 27}, {27, 27}, {28, 27}, {37, 27}, {23, 28}, {27, 28},
        {28, 28}, {29, 28}, {38, 28}, {24, 29}, {28, 29}, {29, 29}, {30, 29},
        {39, 29}, {29, 30}, {30, 30}, {31, 30}, {40, 30}, {30, 31}, {31, 31},
        {32, 31}, {41, 31}, {31, 32}, {32, 32}, {33, 32}, {42, 32}, {32, 33},
        {33, 33}, {34, 33}, {43, 33}, {33, 34}, {34, 34}, {44, 34}, {25, 35},
        {35, 35}, {36, 35}, {45, 35}, {26, 36}, {35, 36}, {36, 36}, {37, 36},
        {46, 36}, {27, 37}, {36, 37}, {37, 37}, {38, 37}, {47, 37}, {28, 38},
        {37, 38}, {38, 38}, {39, 38}, {48, 38}, {29, 39}, {38, 39}, {39, 39},
        {40, 39}, {49, 39}, {30, 40}, {39, 40}, {40, 40}, {41, 40}, {50, 40},
        {31, 41}, {40, 41}, {41, 41}, {42, 41}, {51, 41}, {32, 42}, {41, 42},
        {42, 42}, {43, 42}, {52, 42}, {33, 43}, {42, 43}, {43, 43}, {44, 43},
        {53, 43}, {34, 44}, {43, 44}, {44, 44}, {54, 44}, {35, 45}, {45, 45},
        {46, 45}, {55, 45}, {36, 46}, {45, 46}, {46, 46}, {47, 46}, {56, 46},
        {37, 47}, {46, 47}, {47, 47}, {48, 47}, {57, 47}, {38, 48}, {47, 48},
        {48, 48}, {49, 48}, {58, 48}, {39, 49}, {48, 49}, {49, 49}, {50, 49},
        {59, 49}, {40, 50}, {49, 50}, {50, 50}, {51, 50}, {60, 50}, {41, 51},
        {50, 51}, {51, 51}, {52, 51}, {61, 51}, {42, 52}, {51, 52}, {52, 52},
        {53, 52}, {62, 52}, {43, 53}, {52, 53}, {53, 53}, {54, 53}, {63, 53},
        {44, 54}, {53, 54}, {54, 54}, {64, 54}, {45, 55}, {55, 55}, {56, 55},
        {65, 55}, {46, 56}, {55, 56}, {56, 56}, {57, 56}, {66, 56}, {47, 57},
        {56, 57}, {57, 57}, {58, 57}, {67, 57}, {48, 58}, {57, 58}, {58, 58},
        {59, 58}, {68, 58}, {49, 59}, {58, 59}, {59, 59}, {60, 59}, {69, 59},
        {50, 60}, {59, 60}, {60, 60}, {61, 60}, {70, 60}, {51, 61}, {60, 61},
        {61, 61}, {62, 61}, {71, 61}, {52, 62}, {61, 62}, {62, 62}, {63, 62},
        {72, 62}, {53, 63}, {62, 63}, {63, 63}, {64, 63}, {73, 63}, {54, 64},
        {63, 64}, {64, 64}, {74, 64}, {55, 65}, {65, 65}, {66, 65}, {56, 66},
        {65, 66}, {66, 66}, {67, 66}, {57, 67}, {66, 67}, {67, 67}, {68, 67},
        {58, 68}, {67, 68}, {68, 68}, {69, 68}, {59, 69}, {68, 69}, {69, 69},
        {70, 69}, {60, 70}, {69, 70}, {70, 70}, {71, 70}, {61, 71}, {70, 71},
        {71, 71}, {72, 71}, {62, 72}, {71, 72}, {72, 72}, {73, 72}, {63, 73},
        {72, 73}, {73, 73}, {74, 73}, {64, 74}, {73, 74}, {74, 74} };
}
namespace fs = std::filesystem;
int main(int argc, char *argv[])
{

    std::string training_data_file = "training_data_checkpoint.bin";
    //Checking arguments and creating paths for readability if no arguments are given, reader is run on current directory.
    if (argc > 3) {
        return -1;
    }
    int weight = 0;

    fs::path original_path = (argc == 2) ? fs::path(argv[1]) : fs::path("");
    fs::path train_dir = original_path / MAIN_DIR / "train";
    fs::path test_dir = (original_path / MAIN_DIR / "test" /"pos");

    Training_data* train_data = nullptr;
    if (is_training_data_available(training_data_file)){
        std::cout << "Training data available. \nLoading.." << std::endl;
        train_data = load_training_data(training_data_file);
    }
    else {
        std::cout << "Training data not available. \nTraining new model.." << std::endl;
        train_data = train(train_dir);
        std::cout << "Model has been trained. \nSaving training data to binary file.." << std::endl;
        save_training_data(train_data, training_data_file);
    }

    std::vector<int> review_vector;
    std::ofstream output("op.txt");
    int positives = 0;
    int negatives = 0;
    //BernouliNB evaluation

    float Pc1 = (float)train_data->positive_file_count;
    float Pc0 = (float)train_data->negative_file_count;

    //failsafe laplace smoothing
    double laplace_smoothing = LAPLACEAN_PRIOR / ((train_data->negative_file_count + train_data->positive_file_count) + 2 * LAPLACEAN_PRIOR);

    for (auto const& dir_entry : std::filesystem::directory_iterator(test_dir)) {
        review_vector = get_vector(dir_entry.path(), train_data->guide) ;
        //P(t | c)^x * ( 1-P(t|c) )^(1-x) calculation in log form to avoid underflow
        long double Px1 = log(Pc1); 
        long double  Px0 = log(Pc0);

        for (int index = 0; index < review_vector.size(); ++index) {
            int word_exists = review_vector[index];

            if (word_exists == 0)
                continue;

            //tc = 1 (positive review)
            //tc = 0 (negative review)
            float tc1 = (*train_data->positive)[index];
            float tc0 = (*train_data->negative)[index];

            
          

            //failsafe to avoid 0
            if (tc1 == 0) {
                tc1 = laplace_smoothing;
            }
            if (tc0 == 0) {
                tc0 = laplace_smoothing;
                //std::cout << "tc1: " << tc1 << " tc0: " << tc0 << std::endl;
            }
            // Print statements for debugging
            //std::cout << "Word index: " << index << " tc1: " << tc1 << " tc0: " << tc0 << std::endl;

            Px1 += (word_exists * log(tc1));// +((1 - word_exists) * log(1 - tc1));
            Px0 += (word_exists * log(tc0));// +((1 - word_exists) * log(1 - tc0));

        }
        //std::cout << "Px1: " << Px1 << " Px0: " << Px0 << std::endl;

        float Ppos = exp(Px1);
        float Pneg = exp(Px0);

        //std::cout << "Ppos: " << Ppos << " Pneg: " << Pneg << std::endl;
        if (Px1 > Px0) {
            positives++;
        }
        else {
            negatives++;
        }
        std::string predicted = (Px1 > Px0) ? "Positive" : "Negative";
        std::cout <<  dir_entry << " Predicted : " << predicted << std::endl;
        if (output.is_open()) {
            output << dir_entry << " Predicted : " << predicted << std::endl;
        }
    }
    std::cout << "Positives : " << positives <<" Negatives : " << negatives << std::endl;
    output.close();
    delete train_data;
    return 1;
   
}

