#include "cnn.h"
#include "weights.h"
#include <cmath>

void cnn_inference(float image[28][28], int &predicted_class) {

    float conv_out[8][26][26];
    float pool_out[8][13][13];
    float flat[1352];
    float scores[10];



    predicted_class = /* result */;
}
