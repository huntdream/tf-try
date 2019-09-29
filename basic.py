# import tensorflow package and rename as tf
import tensorflow as tf

# get minist dataset
mnist = tf.keras.datasets.mnist

# load data
(x_train, y_train), (x_test, y_text) = mnist.load_data()

# scale to range [0,1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# build the model with multiple layers
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_text)
