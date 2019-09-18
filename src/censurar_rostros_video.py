import cv2
from operaciones_imagen import censurar_imagen

video = cv2.VideoCapture('video1.mp4')

while True:
    cap, cuadroImagen = video.read()
    imagen_censurada = censurar_imagen(cuadroImagen)
    cv2.imshow('video', imagen_censurada)
    key = cv2.waitKey(1) & 0xFF
    # Para salir del video
    if key == ord("q"):
        break

cv2.destroyAllWindows()
video.stop()