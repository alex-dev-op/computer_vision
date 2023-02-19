import cv2

#GiveImage
img = cv2.imread("example_image.jpg")

#LoadPre-trainedClassifierForDetectingFaces
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#ConvertToGrayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#DetectFaces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#DrawRectangles
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

#Output
cv2.imshow("Detected Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

