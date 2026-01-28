#include <iostream>
#include <cmath>
using namespace std;

int main() {

    int image[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    int kernel[2][2] = {
        {1,  0},
        {0, -1}
    };

    int conv_out[2][2];
    int relu_out[2][2];
    int pool_out[1][1];
    int flat[1];

    for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 2; x++) {
            int sum = 0;
            for (int ky = 0; ky < 2; ky++) {
                for (int kx = 0; kx < 2; kx++) {
                    sum += image[y + ky][x + kx] * kernel[ky][kx];
                }
            }
            conv_out[y][x] = sum;
        }
    }

    for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 2; x++) {
            if (conv_out[y][x] < 0)
                relu_out[y][x] = 0;
            else
                relu_out[y][x] = conv_out[y][x];
        }
    }

    int max_val = relu_out[0][0];
    for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 2; x++) {
            if (relu_out[y][x] > max_val)
                max_val = relu_out[y][x];
        }
    }
    pool_out[0][0] = max_val;

    flat[0] = pool_out[0][0];

    int weights[3][1] = {
        { 1 },
        { 2 },
        { -1 }
    };

    int bias[3] = {0, 1, 3};
    int scores[3];

    for (int c = 0; c < 3; c++) {
        scores[c] = bias[c];
        for (int i = 0; i < 1; i++) {
            scores[c] += flat[i] * weights[c][i];
        }
    }

    double probs[3];
    double sum = 0.0;

    for (int c = 0; c < 3; c++) {
        probs[c] = exp(scores[c]);
        sum += probs[c];
    }

    for (int c = 0; c < 3; c++) {
        probs[c] = probs[c] / sum;
    }

    int predicted_class = 0;
    for (int c = 1; c < 3; c++) {
        if (probs[c] > probs[predicted_class])
            predicted_class = c;
    }

    cout << "Probabilities:" << endl;
    for (int c = 0; c < 3; c++) {
        cout << "Class " << c << ": " << probs[c] << endl;
    }

    cout << "Predicted class: " << predicted_class << endl;

    return 0;
}
