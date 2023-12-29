import geocoder
import cv2
import csv
import datetime
import numpy as np
import os
import tensorflow as tf

def get_live_location():
    try:
        # Get the current location based on IP address
        g = geocoder.ip('me')

        # Check if the location was successfully retrieved
        if g.ok:
            # Extract the latitude and longitude
            latitude = g.lat
            longitude = g.lng

            # Save the location data to a CSV file
            return [latitude, longitude, g.address]

        else:
            print("Location not found")

    except Exception as e:
        print(f"Error: {e}")

def mark_attendance(model_path, dataset_dir, csv_path):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # Load the dataset labels
    label_to_id = {}
    with open(os.path.join(dataset_dir, "labels.txt"), "r") as file:
        labels = file.read().splitlines()
        label_to_id = {label: i for i, label in enumerate(labels)}
    
    # Initialize the CSV file
    csv_file = open(csv_path, 'a')
    csv_writer = csv.writer(csv_file)
    
    # Initialize the webcam
    capture = cv2.VideoCapture(0)
    
    while True:
        # Capture a frame from the webcam
        ret, frame = capture.read()
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Extract the face ROI
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (128, 128))  # Adjust as per the model's input size
            
            # Preprocess the face ROI
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
            face_roi = face_roi / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)
            
            # Predict the label for the face using the trained model
            predictions = model.predict(face_roi)
            predicted_id = np.argmax(predictions)
            predicted_label = labels[predicted_id]
            
            # Mark attendance in the CSV file
            now = datetime.datetime.now()
            date_string = now.strftime("%Y-%m-%d")
            time_string = now.strftime("%H:%M:%S")
            
            # Call the function to get the live location and save the data to a CSV file
            loc_data=get_live_location()

            attendance_data = [predicted_label, time_string, date_string, loc_data]
            #csv_writer.writerow(["Name", "Time", "Date", ['Latitude', 'Longitude', 'Location Address']])
            csv_writer.writerow(attendance_data)
            
            # Draw a rectangle and label around the face in the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Attendance', frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(3000) & 0xFF == ord('q'):
            break

        elif cv2.waitKey(3000) & 0xFF == ord('c'):  # Press 'c' to continue
            continue
    
    # Release the webcam and close the CSV file
    capture.release()
    csv_file.close()
    
    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

    pass

# Mark attendance using the trained model

def main():
    model_path = "trained_model.h5"
    dataset_dir = "Employee_dataset"
    csv_path = "attendance.csv"
    mark_attendance(model_path, dataset_dir, csv_path)

if __name__ == '__main__':
    main()







