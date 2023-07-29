import cv2
import numpy as np
import matplotlib.pyplot as plt

# Wczytanie obrazu
# image = cv2.imread('2.png')
cap = cv2.VideoCapture('klip5.mp4')
ret, frame = cap.read()
paleta = cv2.CascadeClassifier('cascade.xml')

while True:
    # Konwersja do przestrzeni kolorów HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
    # plt.plot(hist)
    # plt.show()

    ret, thresholded_image = cv2.threshold(gray_frame, 180, 255, cv2.THRESH_BINARY)
    cv2.imshow('gray', thresholded_image)

    # Określenie zakresu koloru niebieskiego
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Binaryzacja obrazu
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Odkomentować, aby wyświetlić maskę
    # cv2.imshow('mask', mask)

    # Wykrycie konturów
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Odkomentować, aby wyświetlić kontury
    # cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    # cv2.imshow('Contours', frame)

    # Iteracja przez wszystkie kontury i rysowanie prostokątów
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Ograniczenia testowe
        if 150 < h < 800 and 650 > w > 350:
            disturbance_level = np.mean(thresholded_image[y:y + h, x:x + w]) / 255

            if disturbance_level > 0.01:  # Próg zakłóceń - możesz dostosować wartość według potrzeb
                # Prostokąt czerwony
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, 'pole odkladcze', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                # Prostokąt zielony
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, 'pole odkladcze', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    palety = paleta.detectMultiScale(gray_frame, 1.3, 20, minSize=(200, 70))

    # add this
    for (x, y, w, h) in palety:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(frame, "Paleta", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Wyświetlenie wynikowego obrazu
    cv2.imshow("Wynik", frame)
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnienie obiektu VideoCapture i zamknięcie okna wideo
cap.release()
cv2.destroyAllWindows()
