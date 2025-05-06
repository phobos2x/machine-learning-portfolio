# Import modules
import numpy as np
import os
from scipy.ndimage import rotate, zoom
from numpy.random import rand
from skimage.transform import AffineTransform, warp
from skimage import exposure

def load_data(category, data_dir, max_samples=None):
    # Construct the full path to the data file for a specific category
    path = os.path.join(data_dir, f'full_numpy_bitmap_{category}.npy')

    # Load the data from the NumPy binary file
    data = np.load(path)

    # If max_samples is specified, return only up to that number of samples
    if max_samples is not None:
        return data[:max_samples]

    # If max_samples is not specified, return all loaded data
    return data

def augment_data(data, augment_size=6000):
    # Initialize a list to store augmented data
    augmented_data = []

    for image in data:
        # Stop if the augmented data reaches the specified size
        if len(augmented_data) >= augment_size:
            break

        # Reshape the image to 28x28 for processing
        image = image.reshape(28, 28)

        # Random rotation
        if rand() > 0.5:
            # Rotate the image by a random angle between -15 to 15 degrees
            rotated = rotate(image, angle=np.random.randint(-15, 15), reshape=False)
            augmented_data.append(rotated.reshape(784))

        # Horizontal and vertical flip
        if rand() > 0.5:
            # Flip the image along either the x or y axis
            flipped = np.flip(image, np.random.choice([0, 1]))
            augmented_data.append(flipped.reshape(784))

        # Random zoom
        if rand() > 0.5:
            # Apply random zoom between 0.9 to 1.1 times
            zoom_factor = np.random.uniform(0.9, 1.1)
            zoomed = zoom(image, zoom_factor, order=1)

            # Correct the size of the zoomed image to fit the original dimensions
            zoomed_corrected = np.zeros((28, 28))
            min_row = max((28 - zoomed.shape[0]) // 2, 0)
            min_col = max((28 - zoomed.shape[1]) // 2, 0)
            zoomed_corrected[min_row:min_row + zoomed.shape[0], min_col:min_col + zoomed.shape[1]] = zoomed[:28 - min_row * 2, :28 - min_col * 2]
            augmented_data.append(zoomed_corrected.reshape(784))

        # Shearing
        if rand() > 0.5:
            # Apply a shear transformation with a random shear factor
            af_trans = AffineTransform(shear=np.random.uniform(-0.2, 0.2))
            sheared = warp(image, af_trans, order=1, preserve_range=True, mode='wrap')
            augmented_data.append(sheared.reshape(784))

        # Random brightness adjustment
        if rand() > 0.5:
            # Adjust the brightness of the image within the 0.2nd and 99.8th percentiles
            v_min, v_max = np.percentile(image, (0.2, 99.8))
            brighter = exposure.rescale_intensity(image, in_range=(v_min, v_max))
            augmented_data.append(brighter.reshape(784))

    # Return the augmented data as a NumPy array
    return np.array(augmented_data)

def get_data(categories, data_dir, augment=False):
    # Initialize empty arrays for storing images and labels
    X = np.empty([0, 784])
    Y = np.empty([0])

    # Loop through each category to load and optionally augment data
    for i, category in enumerate(categories):
        # Load the data for the current category
        data = load_data(category, data_dir)

        # If augmentation is enabled, augment the data
        if augment:
            augmented = augment_data(data)
            data = np.concatenate((data, augmented), axis=0)
            augmented_labels = np.full(augmented.shape[0], i)
            labels = np.full(data.shape[0], i)
            Y = np.concatenate((Y, augmented_labels), axis=0)  # Concatenate labels for augmented data
        else:
            labels = np.full(data.shape[0], i)

        # Concatenate the data and labels to the main arrays
        X = np.concatenate((X, data), axis=0)
        Y = np.concatenate((Y, labels), axis=0)

    # Return the combined dataset and corresponding labels
    return X, Y
