#include "test_data.h"



double test_vector(std::vector<int> review_vector, Training_data* model) {

    float Pc0 = (float)model->negative_file_count / (float)(model->positive_file_count + model->negative_file_count); //P(c=0)
    float Pc1 = (float)model->positive_file_count / (float)(model->positive_file_count + model->negative_file_count); //P(c=1)


    long double Px1 = log(Pc1);
    long double Px0 = log(Pc0);

    for (int index = 1; index < review_vector.size() + 1; ++index) {
        int word_exists = review_vector[index];

        if (word_exists == 0)
            continue;

        //tc = 0 (negative review)
        //tc = 1 (positive review)

        float tc0 = (*model->negative_probability_vector)[index];
        float tc1 = (*model->positive_probability_vector)[index];


        // Print statements for debugging
        //std::cout << "Word index: " << index << " tc1: " << tc1 << " tc0: " << tc0 << std::endl;

        //P(t | c)^x * ( 1-P(t|c) )^(1-x) calculation in log form to avoid underflow

        Px1 += (word_exists * log(tc1)) + ((1 - word_exists) * log(1 - tc1));
        Px0 += (word_exists * log(tc0)) + ((1 - word_exists) * log(1 - tc0));

        //result is positive if Px1 > Px0 
        //negative if Px0 > Px1
        return Px1 - Px0;
    }
}
