import cv2

img_path = 'images/test/I_Avenue_2.jpg'
img = cv2.imread(img_path, 1);
cap = cv2.VideoCapture(0)

while(True):
    #cv2.imshow('original', img)
    ret, image_np = cap.read()
    image_np = cv2.resize(image_np, (640, 480))
    cv2.imshow('original', image_np)
    cv2.imwrite('original.png', image_np)
    break
    cv2.waitKey(30)
