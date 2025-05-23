

import cv2
import numpy as np

# Carregar imagem em escala de cinza
img = cv2.imread('imagem_4k.jpg', cv2.IMREAD_GRAYSCALE)

# Binarização (thresholding)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Detecção de componentes conexos
num_labels, labels = cv2.connectedComponents(binary)

print(f"Número de componentes conexos encontrados: {num_labels - 1}")  # Subtrai 1 para ignorar o fundo

# (Opcional) Visualizar os componentes com cores diferentes
output = cv2.applyColorMap((labels * 255 // num_labels).astype(np.uint8), cv2.COLORMAP_JET)
cv2.imwrite("componentes_conexos_output.jpg", output)
