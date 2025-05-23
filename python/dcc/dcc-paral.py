from mpi4py import MPI
import cv2
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Processo 0 lê a imagem completa
if rank == 0:
    img = cv2.imread("imagem_4k.jpg", cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    chunk_size = h // size
    chunks = [img[i*chunk_size:(i+1)*chunk_size if i < size - 1 else h] for i in range(size)]
else:
    chunks = None
    chunk_size = None
    w = None

# Distribuir a largura e o chunk para todos
chunk = comm.scatter(chunks, root=0)

# Cada processo binariza sua parte
_, binary_chunk = cv2.threshold(chunk, 127, 255, cv2.THRESH_BINARY)

# Cada processo detecta componentes na sua parte (local)
_, labels_local = cv2.connectedComponents(binary_chunk)

# Enviar os rótulos locais para o processo 0
gathered_labels = comm.gather(labels_local, root=0)

if rank == 0:
    # Juntar as partes (sem tratar conexões entre blocos!)
    output = np.vstack(gathered_labels)

    print("Imagem processada com MPI. Salvando resultado...")

    # (Opcional) colorir rótulos para visualização
    num_labels = output.max() + 1
    color_output = cv2.applyColorMap((output * 255 // num_labels).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite("componentes_conexos_mpi_output.jpg", color_output)
