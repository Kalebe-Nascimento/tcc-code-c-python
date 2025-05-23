# sudo apt install python3-opencv
#python3 fm-seque.py imagem_4k.jpg


import cv2

# Carregar a imagem
img = cv2.imread('imagem_4k.jpg')

if img is None:
    print("Erro: Imagem não encontrada. Verifique o nome e o caminho.")
    exit()

# Aplicar filtro de mediana (kernel 3x3)
filtered = cv2.medianBlur(img, 3)

# Salvar imagem filtrada
cv2.imwrite('mediana_sequencial_output.jpg', filtered)

print("Imagem processada e salva como 'mediana_sequencial_output.jpg'.")

