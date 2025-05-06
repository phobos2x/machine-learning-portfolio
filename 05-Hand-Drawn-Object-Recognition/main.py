# Import modules
from load_data import get_data
from create_model import create_model
from train_model import prepare_data, train_model
from keras.models import load_model
from keras.optimizers import Adam

# Define the categories for classification
categories = ['bird', 'cat', 'circle', 'house', 'square', 'triangle']

# Specify the directory where the data is located
data_dir = r'C:\Users\Noor\PycharmProjects\Categories'

# Load and augment data for the given categories
X, Y = get_data(categories, data_dir, augment=True)

# Prepare the data for training by splitting into train and test sets
X_train, Y_train, X_test, Y_test = prepare_data(X, Y)

# Create the neural network model with the specified input shape and number of categories
model = create_model((28, 28, 1), len(categories))

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the prepared data
history = train_model(model, X_train, Y_train, X_test, Y_test)

# Load the best performing model saved during training
best_model_path = r'C:\Users\Noor\PycharmProjects\best_model.h5'
best_model = load_model(best_model_path)
