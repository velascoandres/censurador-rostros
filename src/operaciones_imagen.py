import numpy as np
import cv2

# cargar la red neural
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'modelo.caffemodel')


def censurar_imagen(imagen):
    (h, w) = imagen.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(imagen, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Llevar el blob a la red neural para obtener las detecciones y predicciones
    net.setInput(blob)
    detections = net.forward()
    imagen_clonada = imagen.copy()
    for i in range(0, detections.shape[2]):
        # Extraer la conbiabilidad asociada a la deteccion
        confidence = detections[0, 0, i, 2]

        # Filtrar las prediciones debiles en base al nivel de conbiabiliad
        if confidence > 0.55:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, ancho, largo) = box.astype("int")
            rostro = imagen[y:largo, x:ancho]
            rostro_censurado = cv2.GaussianBlur(rostro, (23, 23), 30)
            imagen_clonada[y:y + rostro_censurado.shape[0], x:x + rostro_censurado.shape[1]] = rostro_censurado
    return imagen_clonada
