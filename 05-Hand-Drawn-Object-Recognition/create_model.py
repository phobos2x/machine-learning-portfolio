# Import modules
from keras import layers, Sequential
from keras.regularizers import l2

def create_model(input_shape, num_classes):
    # Create a sequential model, which allows us to build the model layer by layer
    model = Sequential([
        # First convolutional layer with 32 filters, a 3x3 kernel, and ReLU activation.
        # This layer will learn 32 different filters, and it's the input layer of the network.
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),

        # Batch normalization layer to normalize the activations and gradients propagating
        # through the network, which helps in speeding up the training process.
        layers.BatchNormalization(),

        # Max pooling layer with a 2x2 window to reduce the spatial dimensions (height and width)
        # of the output volume.
        layers.MaxPooling2D((2, 2)),

        # Second convolutional layer with 64 filters and ReLU activation.
        # Increasing the number of filters allows the network to learn more complex patterns.
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),

        # Third convolutional layer with 128 filters.
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),

        # Another max pooling layer to further reduce the dimensionality of the feature map.
        layers.MaxPooling2D((2, 2)),

        # Fourth convolutional layer with 256 filters, introducing 'same' padding.
        # Padding allows the layer to apply filters to the edge pixels of the input volume.
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Dropout layer to reduce overfitting by randomly setting a fraction of input units to 0
        # at each update during training time, which helps prevent overfitting.
        layers.Dropout(0.4),

        # Flatten the network to convert 3D feature maps to 1D feature vectors for the dense layers.
        layers.Flatten(),

        # Dense (fully connected) layer with 128 neurons and ReLU activation.
        # Regularization is applied using L2 regularization, which helps prevent overfitting.
        layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),

        # Another dropout layer for regularization.
        layers.Dropout(0.5),

        # Output dense layer with 'num_classes' neurons, one for each class,
        # using softmax activation to output probabilities.
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
