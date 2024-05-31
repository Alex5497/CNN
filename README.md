# CIFAR-10 Classification Using Convolutional Neural Networks

This project demonstrates the classification of the CIFAR-10 dataset using Convolutional Neural Networks (CNNs) with varying depths (1, 2, and 3 convolutional layers). The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. We will be focusing on classifying the images into two broad categories: animals (classes 0-4) and vehicles (classes 5-9).

## Project Structure

- `cifar10_classification.py`: The main script to train and evaluate the models.
- `README.md`: This documentation file.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- scikit-learn

## Setup

1. Install the required libraries:
   ```sh
   pip install tensorflow numpy scikit-learn
   ```

2. Run the script:
   ```sh
   python cifar10_classification.py
   ```

## Script Explanation

### Disable GPU

We start by disabling the GPU to ensure the code runs on CPU. This is useful for environments where GPU is not available.
```python
tf.config.set_visible_devices([], 'GPU')
```

### Load and Preprocess Data

We load the CIFAR-10 dataset and combine the training and test sets into one. We then split this combined dataset into new training and test sets with a ratio of 30:70.
```python
(X_train_full, y_train_full), (X_test_full, y_test_full) = cifar10.load_data()
X_full = np.concatenate((X_train_full, X_test_full), axis=0)
y_full = np.concatenate((y_train_full, y_test_full), axis=0)
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.7, random_state=42)
```

### Binarize Labels

We transform the labels to binary values: 0 for animals and 1 for vehicles.
```python
zwierzeta = [0, 1, 2, 3, 4]
pojazdy = [5, 6, 7, 8, 9]
y_train_bin = np.where(np.isin(y_train, zwierzeta), 0, 1)
y_test_bin = np.where(np.isin(y_test, zwierzeta), 0, 1)
```

### Model Architectures

We define three models with increasing depth of convolutional layers:

1. **Model with one convolutional layer**:
    ```python
    model_1_layer = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    ```

2. **Model with two convolutional layers**:
    ```python
    model_2_layers = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    ```

3. **Model with three convolutional layers**:
    ```python
    model_3_layers = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    ```

### Compile and Train Models

Each model is compiled and trained for 10 epochs.
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_bin, epochs=10, validation_data=(X_test, y_test_bin))
```

### Evaluate Models

The performance of each model is evaluated on the test set.
```python
score = model.evaluate(X_test, y_test_bin)
print(f"Model accuracy: {score[1]}")
```

## Results

After training, the accuracy of each model on the test set is printed.

```python
print(f"Dokładność modelu z jedną warstwą konwolucyjną: {score_1_layer[1]}")
print(f"Dokładność modelu z dwiema warstwami konwolucyjnymi: {score_2_layers[1]}")
print(f"Dokładność modelu z trzema warstwami konwolucyjnymi: {score_3_layers[1]}")
```

## Conclusion

This project demonstrates the impact of model depth on classification performance using CNNs on the CIFAR-10 dataset. By comparing models with different numbers of convolutional layers, we can observe how increasing complexity affects the ability to classify images into animals and vehicles.
