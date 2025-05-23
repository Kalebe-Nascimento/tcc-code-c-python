#sudo apt install mpich
#python3 -m venv venv
#source venv/bin/activate
#pip install mpi4py opencv-python
#mpirun -np 4 python3 fm-paral.py


#fazendo sem ambiente virtual via apt
#sudo apt install python3-opencv python3-mpi4py
#mpirun -np 4 python3 fm-paral.py


from mpi4py import MPI
import cv2
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Processo 0 carrega imagem
if rank == 0:
    img = cv2.imread("imagem_4k.jpg")
    h, w, c = img.shape
    chunk_size = h // size
    chunks = []

    for i in range(size):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < size - 1 else h

        # Adiciona 1 linha de overlap acima e abaixo (se não for a primeira ou última)
        top = max(start - 1, 0)
        bottom = min(end + 1, h)
        chunks.append(img[top:bottom])
else:
    chunks = None
    chunk_size = None
    w = None
    c = None

# Envia pedaços da imagem
chunk = comm.scatter(chunks, root=0)

# Aplica filtro de mediana no pedaço
filtered_chunk = cv2.medianBlur(chunk, 3)

# Remover bordas adicionadas
if rank == 0:
    filtered_chunk = filtered_chunk[0:chunk.shape[0] - 1]
elif rank == size - 1:
    filtered_chunk = filtered_chunk[1:]
else:
    filtered_chunk = filtered_chunk[1:-1]

# Coleta os pedaços processados
gathered = comm.gather(filtered_chunk, root=0)

if rank == 0:
    final_img = np.vstack(gathered)
    cv2.imwrite("mediana_mpi_output.jpg", final_img)
    print("Filtro de mediana com MPI finalizado.")
