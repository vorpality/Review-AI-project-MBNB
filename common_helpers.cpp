#include "common_helpers.h"

std::string completion_bar(int current, int total)
{
    std::string result = " ";
    float percent = (int)((float)(current + 1) / total * 100);
    char empty = '_';
    char filled = '#';

    result += (current > 0) ? filled : empty;
    for (int i = 10; i < 91; i = i + 10) {
        result += (percent > i) ? filled : empty;
    }
    result += (percent == 100) ? filled : empty;
    return result + " " + std::to_string((int)percent) + "%";
}