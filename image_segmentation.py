import cv2

#FunctionForImageSegmentation
def segment_image(image_path, threshold_value):
    #LoadImage
    img = cv2.imread(image_path)

    #ConvertToGrayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #ThresholdImageUsingSpecificThreshValue
    ret, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    #DisplayOriginalAndSegmentedImages
    cv2.imshow("Original Image", img)
    cv2.imshow("Segmented Image", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#EnterInput
image_path = input("Enter the path to the image: ")
threshold_value = int(input("Enter the threshold value (0-255): "))

#CallFunction
segment_image(image_path, threshold_value)
