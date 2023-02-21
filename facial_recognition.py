import cv2

#FacialRecognitionClass
class FaceRecognizer:
    def __init__(self, cascade_file_path):
        #LoadingPre_trainedClassifierFaceDetection
        self.face_cascade = cv2.CascadeClassifier(cascade_file_path)

        #InstanceOfOpenCVFaceRecognitionAlg
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

    def train(self, images, labels):
        #TrainingWithInputImages
        self.recognizer.train(images, labels)

    def recognize_faces(self, image_path):
        #Image
        img = cv2.imread(image_path)

        #ConvertToGrayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #DetectFaces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        #RecognizeAndDrawRectangles
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            label, confidence = self.recognizer.predict(roi_gray)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #Output
        cv2.imshow("Recognized Faces", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#PromptInput
cascade_file_path = input("Enter the path to the face cascade file: ")
images_dir = input("Enter the path to the directory containing the training images: ")
labels_file_path = input("Enter the path to the file containing the training labels: ")
image_path = input("Enter the path to the image to be recognized: ")

#InstanceOfFaceRecognizerClassAndTrainAlg
face_recognizer = FaceRecognizer(cascade_file_path)
images = []
labels = []
for i in range(1, 6):
    img = cv2.imread(f"{images_dir}/person{i}.jpg", 0)
    images.append(img)
    labels.append(i)
face_recognizer.train(images, labels)

#RecognizeFaces
face_recognizer.recognize_faces(image_path)
