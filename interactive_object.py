import cv2

# Define the function to detect objects in an image
def detect_objects(image_path, cascade_file_path):
    # Load the image
    img = cv2.imread(image_path)

    # Load the pre-trained classifier for detecting objects
    object_cascade = cv2.CascadeClassifier(cascade_file_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the objects in the image
    objects = object_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected objects
    for (x, y, w, h) in objects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Detected Objects", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Prompt the user for input
image_path = input("Enter the path to the image: ")
cascade_file_path = input("Enter the path to the cascade file: ")

# Call the detect_objects function with the user's input
detect_objects(image_path, cascade_file_path)
