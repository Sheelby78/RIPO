import numpy as np
import cv2

paleta = cv2.CascadeClassifier('cascade.xml')

cap = cv2.VideoCapture("klip4.mp4")

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    palety = paleta.detectMultiScale(gray, 1.3, 20, minSize=(200, 70))

    # add this
    for (x, y, w, h) in palety:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(img, "Paleta", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
