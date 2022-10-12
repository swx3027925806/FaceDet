import cv2


def Haar(model_path, img_path):
    haar = cv2.CascadeClassifier(model_path)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x + w, y + h), (255,0,0), 2)
    cv2.imshow('dat', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    Haar('model/haarcascade_profileface.xml', 'data\WIDER_val\images\\0--Parade\\0_Parade_marchingband_1_20.jpg')