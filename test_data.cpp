#include "test_data.h"

float test_vector(std::vector<int> review_vector, Training_data* model){

    float Pc0 = (float)model->negative_file_count / (float)(model->positive_file_count + model->negative_file_count); //P(c=0)
    float Pc1 = (float)model->positive_file_count / (float)(model->positive_file_count + model->negative_file_count); //P(c=1)

    //P(t | c)^x * ( 1-P(t|c) )^(1-x) calculation in log form to avoid underflow
    long double Px1 = log(Pc1);
    long double Px0 = log(Pc0);

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