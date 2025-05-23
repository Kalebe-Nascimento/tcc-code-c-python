// filtro_mediana_seq.c
//sudo apt install libopencv-dev pkg-config
//gcc fm-seque.c -o fm-seque `pkg-config --cflags --libs opencv4`


// filtro_mediana_seq.c


#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <stdio.h>

int main() {
    // Carregar imagem colorida
    IplImage* img = cvLoadImage("imagem_4k.png", CV_LOAD_IMAGE_COLOR);
    if (!img) {
        printf("Erro ao carregar imagem.\n");
        return -1;
    }

    // Criar imagem de saída com mesma largura, altura e profundidade
    IplImage* img_saida = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);

    // Aplicar filtro de mediana com tamanho da janela 3x3
    cvSmooth(img, img_saida, CV_MEDIAN, 3, 3, 0, 0);

    // Salvar imagem de saída
    if (!cvSaveImage("saida_mediana_seq.png", img_saida)) {
        printf("Erro ao salvar imagem de saída.\n");
    } else {
        printf("Filtro de mediana aplicado com sucesso (sequencial).\n");
    }

    // Liberar imagens
    cvReleaseImage(&img);
    cvReleaseImage(&img_saida);

    return 0;
}
