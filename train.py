#pip install tensorflow numpy


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


x_train = x_train.astype(np.float32) / 255.0
x_test  = x_test.astype(np.float32) / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

model = models.Sequential([
    layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=20, batch_size=64)


test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)



conv_weights, conv_bias = model.layers[0].get_weights()

fc_weights, fc_bias = model.layers[3].get_weights()

np.savetxt("conv_weights.txt", conv_weights.reshape(-1), fmt="%.6f")
np.savetxt("conv_bias.txt", conv_bias, fmt="%.6f")

np.savetxt("fc_weights.txt", fc_weights.reshape(-1), fmt="%.6f")
np.savetxt("fc_bias.txt", fc_bias, fmt="%.6f")

print("Weights exported successfully.")
