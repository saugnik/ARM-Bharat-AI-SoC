#include <iostream>
#include "cnn.h"

int main() {
    float image[28][28];
    int predicted_class;


    cnn_inference(image, predicted_class);

    std::cout << "Predicted digit: " << predicted_class << std::endl;
    return 0;
}
