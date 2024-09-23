import tensorflow as tf
import matplotlib.pyplot as plt

# MNIST-Daten laden
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Funktion zur Erstellung eines Modells mit einer bestimmten Aktivierungsfunktion
def create_model(activation_function):
    if activation_function == 'leaky_relu':
        activation_layer = tf.keras.layers.LeakyReLU()
    else:
        activation_layer = activation_function
    
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation=activation_layer),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Testen verschiedener Aktivierungsfunktionen
activation_functions = ['relu', 'sigmoid', 'tanh', 'leaky_relu']
histories = {}

for func in activation_functions:
    model = create_model(activation_function=func)
    print(f"Training with {func} activation function...")
    history = model.fit(x_train, y_train, epochs=64, validation_data=(x_test, y_test), verbose=0)
    histories[func] = history

# Plotten der Ergebnisse
plt.figure(figsize=(66, 6))

for func in activation_functions:
    plt.plot(histories[func].history['val_accuracy'], label=func)

plt.title('Validation Accuracy by Activation Function')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()
