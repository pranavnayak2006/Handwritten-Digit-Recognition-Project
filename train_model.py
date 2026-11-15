
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

print("HANDWRITTEN DIGIT RECOGNITION")
print("--------------------------------")

# Step 1: Create models folder
os.makedirs("models", exist_ok=True)

# Step 2: Load MNIST dataset
print("Loading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)

# Step 3: Normalize image pixels
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Step 4: Reshape images for CNN
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# Step 5: Create CNN model
print("Building CNN model...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128, activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        10, activation="softmax"
    )
])

# Step 6: Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Step 7: Display model summary
model.summary()

# Step 8: Train the model
print("Starting model training...")

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# Step 9: Evaluate the model
print("Evaluating model...")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test
)

print("--------------------------------")
print("FINAL RESULTS")
print("Test Accuracy:", test_accuracy * 100, "%")
print("Test Loss:", test_loss)

# Step 10: Save the trained model
model.save("models/digit_model.keras")

print("Model saved successfully!")

# Step 11: Save accuracy graph
plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"],
         label="Training Accuracy")

plt.plot(history.history["val_accuracy"],
         label="Validation Accuracy")

plt.title("CNN Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy.png")
plt.close()

# Step 12: Save loss graph
plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"],
         label="Training Loss")

plt.plot(history.history["val_loss"],
         label="Validation Loss")

plt.title("CNN Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss.png")
plt.close()

print("Accuracy and loss graphs saved!")
print("TRAINING COMPLETED!")