#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#define MAX_LINE_LENGTH 1024 
#define NUM_THREADS 12

typedef struct {
    char*** matriz;
    int linhas;
    int colunas;
} MatrizInfo;


MatrizInfo calc_size_data(char* filepath) {
    MatrizInfo data;
    char str[MAX_LINE_LENGTH];
    int result;
    FILE* file = fopen(filepath, "r");
    int n_cols = 1;
    char *token;
    char line[MAX_LINE_LENGTH];

    int linha = 0;
    int col = 0;

   
   while (fgets(line, sizeof(line), file) != NULL) {
        line[strcspn(line, "\n")] = '\0';
        token = strtok(line, ",");

        col = 0;
        while (token != NULL) {
            if (token[0] == '\0') {
                for (int i = 0; i <= col; i++) {
                    free(data.matriz[linha][i]); 
                }
                col = 0; 
                break;
            }

            
            token = strtok(NULL, ","); 
            col++;
        }

        linha++;
    }
    col++;
    // Alocar dinamicamente a matriz de strings
    char*** matriz = (char***)malloc(linha * sizeof(char**));
    for (int i = 0; i < linha; i++) {
        matriz[i] = (char**)malloc(col * sizeof(char*));
    }

    // Preencher a MatrizInfo
    data.matriz = matriz;
    data.linhas = linha;
    data.colunas = col;

    fclose(file);

    return data;
}


MatrizInfo get_data(char* filepath) {
    MatrizInfo data = calc_size_data(filepath);
    char* str_centroiude = "centroide";
    for(int i = 0;i<data.linhas;i++){
        data.matriz[i][data.colunas-1] = strdup(str_centroiude);// definindo uma coluna extra para o valor do centroide
    }
    char str[MAX_LINE_LENGTH];
    int result;
    FILE *file = fopen(filepath, "r"); 

    if (file == NULL) {
        printf("Erro ao abrir o arquivo.\n");
    }

    char line[MAX_LINE_LENGTH];
    char *token;

    int linha = 0;
    int col = 0;


    while (fgets(line, sizeof(line), file) != NULL) {
        line[strcspn(line, "\n")] = '\0';
        token = strtok(line, ",");

        col = 0;
        while (token != NULL) {
            data.matriz[linha][col] = strdup(token);

            if (data.matriz[linha][col][0] == '\0') {
                for (int i = 0; i <= col; i++) {
                    free(data.matriz[linha][i]); 
                }
                col = 0; 
                break; 
            }

            token = strtok(NULL, ","); 
            col++;
        }

        if (col < data.colunas-1) {
            // A linha não tem o número esperado de campos, então é inválida
            for (int i = 0; i < col; i++) {
                free(data.matriz[linha][i]); // Libera a memória das strings já alocadas
            }
            continue; // Vai para a próxima iteração (próxima linha)
        }

        linha++;
    }
    //atualiza quantidade de linhas para remover as faltantes
    data.linhas = linha;
    fclose(file);
    return data;
}


//encontra o menor valor de forma sequencia
double min_value(MatrizInfo* data, int col){
    double menor_valor = atof(data->matriz[1][col]);
    //uso do reduction para garantir que na condição as threads não atualizem a variavel menor_valor, a todo momento com valores locais
    #pragma omp parallel for reduction(min:menor_valor) num_threads(NUM_THREADS) 
        for(int i = 1; i< data->linhas;i++){
            double value = atof(data->matriz[i][col]);
            if(value < menor_valor){
                menor_valor = value;
            }
        }
    
    return menor_valor;
}


double max_value(MatrizInfo* data, int col){
    double maior_valor = atof(data->matriz[1][col]);
    //uso do reduction para garantir que na condição as threads não atualizem a variavel maior_valor, a todo momento com valores locais

    #pragma omp parallel for reduction(max:maior_valor) num_threads(NUM_THREADS)
        for(int i = 1; i< data->linhas;i++){
            if(atof(data->matriz[i][col]) > maior_valor){
                    maior_valor = atof(data->matriz[i][col]);
           }
        }

    return maior_valor;
}


int gerarValorAleatorio(int min, int max) {
    return min + rand() % (max - min + 1);
}


void free_matriz(MatrizInfo info) {
    for (int i = 0; i < info.linhas; i++) {
        for (int j = 0; j < info.colunas; j++) {
            free(info.matriz[i][j]);
        }
        free(info.matriz[i]);
    }
    free(info.matriz);
}


void get_euclidian_distance(MatrizInfo* data, MatrizInfo* centroides, int *indices, int n, int linha) {
    double dist = 0.0;
    double menor_dist = 0.0;
    int best_centroid = 0;
    for (int ii = 1; ii < centroides->linhas; ii++) {
        for (int i = 0; i < n; i++) {
            int index = indices[i];
            double value_data = atof(data->matriz[linha][index]);
            double value_centroid = atof(centroides->matriz[ii][index]);
            double difference = value_data - value_centroid;
            dist = dist + pow(difference, 2);
        }
        dist = sqrt(dist);
        if(ii == 1){
            menor_dist = dist;
            best_centroid = ii;
        }
        if (dist < menor_dist) {
            menor_dist = dist;
            best_centroid = ii;
        }
        dist = 0;
    }

    // Convertendo o valor do best_centroid para uma string
    char best_centroid_str[11];
    sprintf(best_centroid_str, "%d", best_centroid);

    data->matriz[linha][data->colunas - 1] = strdup(best_centroid_str);
}


void update_centroides(MatrizInfo* data, MatrizInfo* centroides, int* indices, int n) {
    int centroid_atr = -1;
    int* qtd = (int*)calloc(centroides->linhas, sizeof(int)); 
    //aloca uma matriz extra para evitar acessos demais de escrita e leitura na matriz original
    double** soma_parcial = (double**)malloc(centroides->linhas * sizeof(double*));

    for (int i = 0; i < centroides->linhas; i++) {
        soma_parcial[i] = (double*)calloc(data->colunas, sizeof(double));
    }

    // Parte paralela da atualização inicial dos centroides
    // private(centroid_art) para protejar a variavel de concorrencia  e shared da soma_parcila e qtd para que todas as threads trabalhem no mesmo conjunto
    #pragma omp parallel for private(centroid_atr) shared(soma_parcial, qtd)
    for (int i = 1; i < data->linhas; i++) {
        centroid_atr = atoi(data->matriz[i][data->colunas - 1]);
        //atualizar os valores compartilhados de modo que somente 1thread faça isso por vez
        #pragma omp atomic
        qtd[centroid_atr]++;
        for (int ii = 0; ii < n; ii++) {
            int index = indices[ii];
            double dado = atof(data->matriz[i][index]);
            //atualizar os valores compartilhados de modo que somente 1thread faça isso por vez
            #pragma omp atomic
            soma_parcial[centroid_atr][index] += dado;
        }
    }

    // Calcular e atualizar os centroides paralelizando o laço externo
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 1; i < centroides->linhas; i++) {
        for (int ii = 0; ii < n; ii++) {
            int index = indices[ii];
            double dado_f = soma_parcial[i][index] / qtd[i];
            char dado_str[1024];
            sprintf(dado_str, "%.4f", dado_f);
            centroides->matriz[i][index] = strdup(dado_str);
        }
        free(soma_parcial[i]);
    }

    free(soma_parcial);
    free(qtd);
}


void norm_data(MatrizInfo* info, int* indices, int n) {
    
    for (int ii = 0; ii < n; ii++) {
        int n_col = indices[ii];
        double min = min_value(info, n_col);
        double max = max_value(info, n_col);
        char string_numero[1024];
        double mat_value = 0.0;
        double norm_value = 0.0;
        //paralelizando o laço de normalizaçao de forma simples para normalizar linha a linha de cada coluna simultaneamente
        #pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = 1; i < info->linhas; i++) {
            double mat_value = atof(info->matriz[i][n_col]);
            double norm_value = ((mat_value - min) / (max - min));
            char string_numero[1024];
            sprintf(string_numero, "%.4f", norm_value);

            #pragma omp atomic write//atualiza na matriz somente 1 thread por vez
            info->matriz[i][n_col] = strdup(string_numero);
        }
    }
}


void init_centers(MatrizInfo* data, MatrizInfo* center, int n_centers) {
    n_centers = n_centers + 1; // Adiciona uma linha para a label

    char*** matriz = (char***)malloc(n_centers * sizeof(char**));
    for (int i = 0; i < n_centers; i++) {
        matriz[i] = (char**)malloc(data->colunas * sizeof(char*));
    }

    for (int i = 0; i < n_centers; i++) {
        for (int ii = 0; ii < data->colunas; ii++) {
            if (i == 0) {
                matriz[i][ii] = data->matriz[0][ii]; // Label das colunas
            } else {
                char string_numero[1024];

                int index_in_data = gerarValorAleatorio(1, data->linhas);
                int index_in_data2 = gerarValorAleatorio(1, data->linhas);
            
                double v1 = atof(data->matriz[index_in_data][ii]);
                double v2 = atof(data->matriz[index_in_data2][ii]);
                double v = (v1+v2)/2;
                sprintf(string_numero, "%.4f", v);
                matriz[i][ii] = strdup(string_numero) ; // Dados
            }
        }
    }

    center->matriz = matriz;
    center->colunas = data->colunas;
    center->linhas = n_centers;
}


void init_centers_def(MatrizInfo* data, MatrizInfo* center, int n_centers) {
    n_centers = n_centers + 1; // Adiciona uma linha para a label

    char*** matriz = (char***)malloc(n_centers * sizeof(char**));
    for (int i = 0; i < n_centers; i++) {
        matriz[i] = (char**)malloc(data->colunas * sizeof(char*));
    }

    for (int i = 0; i < n_centers; i++) {
        for (int ii = 0; ii < data->colunas; ii++) {
            if (i == 0) {
                matriz[i][ii] = data->matriz[0][ii]; // Label das colunas
            } else {
                char string_numero[1024];

                
                int index_in_data = i*7;
                int index_in_data2 = i*3;
                double v1 = atof(data->matriz[index_in_data][ii]);
                double v2 = atof(data->matriz[index_in_data2][ii]);
                double v = (v1+v2)/2;
                sprintf(string_numero, "%.4f", v1);
                matriz[i][ii] = strdup(string_numero) ; // Dados
            }
        }
    }

    center->matriz = matriz;
    center->colunas = data->colunas;
    center->linhas = n_centers;
}


void write_csv(const char* filepath, MatrizInfo* matriz) {
    FILE* f = fopen(filepath, "w");
    if (f == NULL) {
        printf("Erro ao abrir o arquivo.\n");
        return;
    }

    for (int i = 0; i < matriz->linhas; i++) {
        for (int j = 0; j < matriz->colunas; j++) {
            fprintf(f, "%s", matriz->matriz[i][j]);
            if (j < matriz->colunas - 1) {
                fprintf(f, ",");
            }
        }
        fprintf(f, "\n");
    }

    fclose(f);
}


void print_matrix(MatrizInfo* matriz, int n_linhas){
    for (int i = 0; i < n_linhas ; i++) {
        printf("%d : ",i);
        for (int j = 0; j < matriz->colunas; j++) {
            printf(" %s ", matriz->matriz[i][j]);
        }
        printf("\n");
    }
}


void print_progress_bar(int progress) {
    int bar_length = 50;
    int filled_length = bar_length * progress / 100;

    printf("\r[");
    for (int i = 0; i < bar_length; i++) {
        if (i < filled_length) {
            putchar('#');
        } else {
            putchar(' ');
        }
    }
    printf("] %3d%%", progress);
    fflush(stdout);
}


int main() {
    double inicio, fim;
    inicio = omp_get_wtime();

    char* filepath = "/home/gustavo/PersonalProjects/K-Means-torpedo/dataset/housing.csv";
    printf("Collecting data \n");
    MatrizInfo data = get_data(filepath);
    printf("Data shape %d x %d \n", data.linhas, data.colunas);
    printf("Defining centroids \n");
    MatrizInfo centroids;
    int n_centroides = 4;
    int n = 9;
    //int indices[3] = {0,1,7};
    int indices[n];
    for(int i=0;i<n;i++){
        indices[i] = i;
    }
    printf("Normalizing data \n");
    norm_data(&data, indices, n);
    print_matrix(&data,5);
    printf("Init random centroids \n");

    init_centers(&data,&centroids ,n_centroides);
    print_matrix(&centroids, centroids.linhas);
    //kmeans loop
    int n_iters = 10;
    printf("K means parallel running for %d iters\n", n_iters);
    for(int iter=0;iter<n_iters;iter++){
        int progress = (iter + 1) * 100 / n_iters;
        print_progress_bar(progress);
        //pareleliza loop interno do calculo de distancias para calcular varias distancias ao mesmo tempo, usa o static, para dividir 
        // as iterações de cada chamada da função get_euclidian_distance de forma igual entre as thread
        #pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
        for(int linha=1; linha < data.linhas; linha++){
            get_euclidian_distance(&data, &centroids, indices, n, linha);
        }
        update_centroides(&data,&centroids, indices, n);

    }
    printf("\n Writing matrix with results \n");
    write_csv("kmeans_results_par.csv", &data);

    printf("\n Writing matrix with centroids \n");
    write_csv("kmeans_centroids_par.csv", &centroids);
    free_matriz(data);
    
    fim = omp_get_wtime();           //funcao omp de tempo
    printf("\n\nTempo total: %.6f segundos.\n", fim-inicio);
    return 0;
}
