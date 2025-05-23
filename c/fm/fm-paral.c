// filtro_mediana_mpi.c
#include <opencv2/opencv.hpp>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cv::Mat img;
    int rows, cols, channels;

    if (rank == 0) {
        img = cv::imread("imagem_4k.png", cv::IMREAD_COLOR);
        if (img.empty()) {
            printf("Erro ao carregar imagem.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        rows = img.rows;
        cols = img.cols;
        channels = img.channels();
    }

    // Broadcast dimensões
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_rows = rows / size;
    int extra = rows % size;
    if (rank == size - 1) local_rows += extra;

    // Borda superior e inferior (+2)
    int buffer_rows = local_rows + 2;
    uchar *recv_data = (uchar *)malloc(buffer_rows * cols * channels);

    if (rank == 0) {
        for (int i = 0, offset = 0; i < size; i++) {
            int count = rows / size;
            if (i == size - 1) count += extra;

            int start = (i == 0) ? 0 : offset - 1;
            int send_rows = count + ((i == 0 || i == size - 1) ? 1 : 2);

            MPI_Send(img.ptr<uchar>(start), send_rows * cols * channels, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
            offset += count;
        }
    }

    // Recebe dados com bordas
    int recv_rows = buffer_rows;
    if (rank == 0) recv_rows--;
    if (rank == size - 1) recv_rows--;

    MPI_Recv(recv_data, recv_rows * cols * channels, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Converte para Mat
    cv::Mat input(recv_rows, cols, CV_8UC3, recv_data);
    cv::Mat output;
    cv::medianBlur(input, output, 3);

    // Remove bordas antes de enviar ao rank 0
    int valid_start = (rank == 0) ? 0 : 1;
    int valid_end = (rank == size - 1) ? output.rows : output.rows - 1;
    cv::Mat valid = output.rowRange(valid_start, valid_end);

    // Envia resultado ao rank 0
    if (rank != 0) {
        MPI_Send(valid.data, valid.rows * cols * channels, MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD);
    } else {
        cv::Mat final(rows, cols, CV_8UC3);
        valid.copyTo(final.rowRange(0, valid.rows));

        int offset = valid.rows;
        for (int i = 1; i < size; i++) {
            int recv_count = rows / size;
            if (i == size - 1) recv_count += extra;

            cv::Mat part(recv_count, cols, CV_8UC3);
            MPI_Recv(part.data, recv_count * cols * channels, MPI_UNSIGNED_CHAR, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            part.copyTo(final.rowRange(offset, offset + recv_count));
            offset += recv_count;
        }

        cv::imwrite("saida_mediana_mpi.png", final);
        printf("Filtro de mediana aplicado com MPI.\n");
    }

    free(recv_data);
    MPI_Finalize();
    return 0;
}
