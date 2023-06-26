import cv2

img=cv2.imread('whitehat Project/p-106/PRO-106-ProjectTemplate-main/PRO-106-ProjectTemplate-main/walking.avi')

# Create our body classifier
body_classifier=cv2.CascadeClassifier('whitehat Project/p-106/PRO-106-ProjectTemplate-main/PRO-106-ProjectTemplate-main/haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=body_classifier.detectMultiScale(gray)
    for (x,y,w,h) in img:
       cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    #Convert Each Frame into Grayscale
    
    # Pass frame to our body classifier
    bodies=body_classifier.detectMultiScale(gray,1.2,3)
    
    # Extract bounding boxes for any bodies identified
    cv2.imshow("gray", frame)

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
