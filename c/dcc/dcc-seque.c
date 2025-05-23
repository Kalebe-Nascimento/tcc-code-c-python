// componentes_conexos_seq.c
#include <opencv2/opencv.hpp>
#include <stdio.h>

int main() {
    cv::Mat img = cv::imread("imagem_4k.png", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        printf("Erro ao carregar imagem.\n");
        return -1;
    }

    // Binariza
    cv::Mat bin;
    cv::threshold(img, bin, 127, 255, cv::THRESH_BINARY);

    // Aplica componentes conexos
    cv::Mat labels;
    int n_labels = cv::connectedComponents(bin, labels);

    printf("Número de componentes conexos: %d\n", n_labels - 1);

    // Visualização colorida
    cv::Mat colorMap;
    labels.convertTo(colorMap, CV_8U, 255.0 / n_labels);
    cv::applyColorMap(colorMap, colorMap, cv::COLORMAP_JET);
    cv::imwrite("componentes_conexos_seq_output.png", colorMap);

    return 0;
}
