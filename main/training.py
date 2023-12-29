import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers



def create_labels_file(dataset_dir):                #Creates labels.txt file
    labels = os.listdir(dataset_dir)
    
    with open(os.path.join(dataset_dir, 'labels.txt'), 'w') as file:
        file.write('\n'.join(labels))
    
    pass





def train_model(dataset_dir):
    # Load the dataset
    image_size = (128, 128)  # Adjust as per your requirements
    data = []
    labels = []

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                label = os.path.basename(root)
                
                # Load and preprocess the image
                img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
                img = tf.keras.preprocessing.image.img_to_array(img)
                img = img / 255.0  # Normalize pixel values between 0 and 1
                
                data.append(img)
                labels.append(label)
    
    data = np.array(data)
    labels = np.array(labels)
    
    # Create a mapping of labels to numerical values
    label_to_id = {label: i for i, label in enumerate(np.unique(labels))}
    id_to_label = {i: label for label, i in label_to_id.items()}
    
    # Convert labels to numerical values
    labels = np.array([label_to_id[label] for label in labels])
    
    # Split the dataset into training and validation sets
    num_classes = len(label_to_id)
    (train_data, train_labels), (val_data, val_labels) = split_dataset(data, labels, test_size=0.2)
    
    # Create the model
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    # Train the model
    model.fit(train_data, train_labels, epochs=10, validation_data=(val_data, val_labels))
    
    # Save the trained model
    model.save("trained_model.h5")
    
    print("Model training completed and saved.")
    pass


def split_dataset(data, labels, test_size=0.2):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    split = int(test_size * data.shape[0])

    train_indices = indices[split:]
    test_indices = indices[:split]

    train_data = data[train_indices]
    train_labels = labels[train_indices]
    val_data = data[test_indices]
    val_labels = labels[test_indices]

    return (train_data, train_labels), (val_data, val_labels)







# Train the model using the collected dataset
def main():
    dataset_dir = "Employee_dataset"    # Path to the dataset directory
    train_model(dataset_dir)
    create_labels_file(dataset_dir)     # Create labels.txt file in the dataset directory

if __name__ == '__main__':
    main()