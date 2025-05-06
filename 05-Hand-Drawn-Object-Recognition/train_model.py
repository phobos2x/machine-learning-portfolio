# Import modules
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import numpy as np

def prepare_data(X, Y, test_size=10000):
    # Set a random seed for reproducibility
    np.random.seed(1)

    # Shuffle the dataset to ensure a random distribution of data
    shuffle_indices = np.random.permutation(np.arange(X.shape[0]))
    X = X[shuffle_indices]
    Y = Y[shuffle_indices]

    # Split the dataset into training and testing sets
    X_test, Y_test = X[:test_size], Y[:test_size]
    X_train, Y_train = X[test_size:], Y[test_size:]

    # Reshape and normalize the data for the neural network
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1) / 255.0

    # Convert the labels to one-hot encoded vectors
    Y_train = to_categorical(Y_train, len(np.unique(Y)))
    Y_test = to_categorical(Y_test, len(np.unique(Y)))

    # Return the prepared data
    return X_train, Y_train, X_test, Y_test

def train_model(model, X_train, Y_train, X_test, Y_test, epochs=30, batch_size=32):
    # Define callbacks for model training
    checkpoint = ModelCheckpoint("best_model.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)

    # Compile the model with Adam optimizer and categorical crossentropy loss
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model on the training data and validate on the test data
    return model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_test, Y_test), batch_size=batch_size,
                     callbacks=[checkpoint, early_stopping, reduce_lr])
