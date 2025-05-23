// componentes_conexos_mpi.c
#include <opencv2/opencv.hpp>
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cv::Mat img;
    int rows, cols;

    if (rank == 0) {
        img = cv::imread("imagem_4k.png", cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            printf("Erro ao carregar imagem.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        rows = img.rows;
        cols = img.cols;
    }

    // Envia dimensões para todos
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Divide por linhas
    int local_rows = rows / size;
    int remainder = rows % size;
    if (rank == size - 1) local_rows += remainder;

    uchar *local_data = (uchar *)malloc(local_rows * cols * sizeof(uchar));

    // Scatter linhas
    if (rank == 0) {
        for (int i = 0, offset = 0; i < size; i++) {
            int count = rows / size;
            if (i == size - 1) count += remainder;
            MPI_Send(img.ptr<uchar>(offset), count * cols, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
            offset += count;
        }
    }

    MPI_Recv(local_data, local_rows * cols, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Processa local
    cv::Mat local_img(local_rows, cols, CV_8UC1, local_data);
    cv::Mat binary_local, labels_local;
    cv::threshold(local_img, binary_local, 127, 255, cv::THRESH_BINARY);
    int n_labels = cv::connectedComponents(binary_local, labels_local);

    printf("Rank %d detectou %d componentes (não unificados).\n", rank, n_labels - 1);

    // Envia para o rank 0
    if (rank != 0) {
        MPI_Send(labels_local.data, local_rows * cols * sizeof(int), MPI_BYTE, 0, 1, MPI_COMM_WORLD);
    } else {
        cv::Mat full_labels(rows, cols, CV_32S);
        labels_local.copyTo(full_labels.rowRange(0, local_rows));
        for (int i = 1, offset = local_rows; i < size; i++) {
            int count = rows / size;
            if (i == size - 1) count += remainder;
            MPI_Recv(full_labels.ptr<int>(offset), count * cols, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            offset += count;
        }

        // Converte e colore
        cv::Mat out;
        full_labels.convertTo(out, CV_8U, 255.0 / (n_labels + 1));
        cv::applyColorMap(out, out, cv::COLORMAP_JET);
        cv::imwrite("componentes_conexos_mpi_output.png", out);
    }

    free(local_data);
    MPI_Finalize();
    return 0;
}
