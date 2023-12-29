# Employee Attendance System

This project implements an Employee Attendance System using facial recognition. The system involves capturing a dataset of employees' facial images, training a convolutional neural network (CNN) to recognize these faces, and using the trained model to mark attendance based on live webcam input.

## Setup

### Requirements

Ensure you have Python installed. You can install the required packages using the following command:


### Dataset Collection

To create the dataset, run the `capture_dataset.py` script. This script captures a specified number of images for each employee. Execute it with the following command:


### Train the Model

To train the facial recognition model, use the `train_model.py` script. This script loads the dataset, preprocesses the images, and trains a CNN using TensorFlow. Run it as follows:


### Attendance Marking

Once the model is trained, attendance can be marked using the `mark_attendance.py` script. This script uses the trained model to recognize employees in a live webcam feed and marks their attendance. Execute it as follows:


## Additional Features

### Live Location

The system includes a feature to get the live location of the employee based on their IP address. This is achieved using the `geocoder` library.

### Dataset Organization

The dataset is organized with each employee's images in a separate directory. The `create_labels_file` function creates a `labels.txt` file, mapping each employee's directory to a unique label.

## Note

- Ensure that the `Pillow` library is installed to handle image processing. Install it using:


- Make sure to have a webcam connected for dataset collection and attendance marking.

Feel free to customize the project based on your requirements.
