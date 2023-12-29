import cv2
import os
import time

def capture_dataset(num_users, images_per_user):
    # Create a new directory to save the dataset
    dataset_dir = "Employee_dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    
    for user_id in range(1, num_users + 1):
        # Create a new directory for the user
        employee_name = input(f"Enter the EmpID of User {user_id}: ")
        user_dir = os.path.join(dataset_dir, employee_name)
        os.makedirs(user_dir, exist_ok=True)
       
        print(f"Capturing images for User {user_id}")
        
        for image_num in range(1, images_per_user + 1):
            print("Wait for 2 seconds for new capture")
            time.sleep(2)
            # Open the camera
            capture = cv2.VideoCapture(0)
            
            # Capture a frame from the camera
            ret, frame = capture.read()
            
            # Check if the frame was successfully captured
            if not ret:
                print("Failed to capture image")
                continue
            
            # Save the captured image
            image_path = os.path.join(user_dir, f"image_{image_num}.jpg")
            cv2.imwrite(image_path, frame)
            
            print(f"Saved image {image_num}/{images_per_user} for User {user_id}")
            
            # Release the camera
            capture.release()
            
    print("Dataset collection completed")

    pass

def main():
    num_users=int(input("Enter number of Employees: "))
    images_per_user = 5
    capture_dataset(num_users, images_per_user)

if __name__ == '__main__':
    main()