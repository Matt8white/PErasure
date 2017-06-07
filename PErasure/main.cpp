extern "C"{
	#include "gf_rand.h"
}
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "jerasure.h"

using namespace std;

#define talloc(type, num) (type *) malloc(sizeof(type)*(num))

static void print_data_and_coding(int k, int m, int w, int size, char **data, char **coding) {
    int i, j, x;
    int n, sp;
    
    if(k > m) n = k;
    else n = m;
    sp = size * 2 + size/(w/8) + 8;
    
    printf("%-*sCoding\n", sp, "Data");
    for(i = 0; i < n; i++) {
        if(i < k) {
            printf("D%-2d:", i);
            for(j=0;j< size; j+=(w/8)) {
                printf(" ");
                for(x=0;x < w/8;x++){
                    printf("%02x", (unsigned char)data[i][j+x]);
                }
                }
            printf("    ");
        }
        else printf("%*s", sp, "");
        if(i < m) {
            printf("C%-2d:", i);
            for(j=0;j< size; j+=(w/8)) {
                printf(" ");
                for(x=0;x < w/8;x++){
                    printf("%02x", (unsigned char)coding[i][j+x]);
                }
            }
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv){
	
	unsigned int m, k, w, i, j, seed, psize;
	int *matrix;
	int *bitmatrix;
    clock_t start;
	char **data, **coding;
    srand((unsigned int)time(NULL));
    
    if(argc != 5) {
        fprintf(stderr, "Please add arguments k, m, w and packetsize\n");
        exit(1);
    }
	if(sscanf(argv[1], "%d", &k) == 0 || k <= 0) {
		fprintf(stderr, "Wrong k\n"); 
		exit(1);
	}
	if (sscanf(argv[2], "%d", &m) == 0 || m <= 0) {
		fprintf(stderr, "Wrong m\n"); 
		exit(1);
	}
	if (sscanf(argv[3], "%d", &w) == 0 || w <= 0 || w > 31) {
		fprintf(stderr, "Wrong w\n"); 
		exit(1);
	}
	if (sscanf(argv[4], "%d", &psize) == 0) {
		fprintf(stderr, "Wrong packetsize\n"); 
		exit(1);
	}
	if((k + m) > (1 << w)) {
		fprintf(stderr, "Wrong w, the following must hold: m + k <= 2^w\n"); 
		exit(1);
	}
    
//    Creating matrix and BDM
    seed = rand();
    MOA_Seed(seed);
    matrix = talloc(int, m*k);
    for (i = 0; i < m; i++) {
        for (j = 0; j < k; j++) {
            matrix[i*k+j] = galois_single_divide(1, i ^ (m + j), w);
        }
    }
    bitmatrix = jerasure_matrix_to_bitmatrix(k, m, w, matrix);

//    Generating fake random data
	data = talloc(char *, k);
	for (i = 0; i < k; i++) {
		data[i] = talloc(char, psize*w);
		MOA_Fill_Random_Region(data[i], psize*w);
	}

	coding = talloc(char *, m);
	for (i = 0; i < m; i++) {
		coding[i] = talloc(char, psize*w);
	}
    start = clock();
    jerasure_bitmatrix_encode(k, m, w, bitmatrix, data, coding, w*psize, psize);
    printf("Encoding Complete, time elapsed: %.4fs\n", (clock() - (float)start) / CLOCKS_PER_SEC);
    //print_data_and_coding(k, m, w, psize, data, coding);
    printf("\n");
    
//    Erasing m devices
    int random[m+1], r;
    bool flag;
    for(i = 0; i < m;) {
        r = MOA_Random_W(w, 1) % (k + m);
        flag = true;
        for (j = 0; j < m; j++)
            if (r == random[j]) flag = false;
        if (flag) {
            random[i] = r;
            i++;
        }
    }
    random[i] = -1;
    for(i = 0; i < m; i++) {
        if (random[i] < k)
            bzero(data[random[i]], w*psize);
        else bzero(coding[random[i] - k], w*psize);
    }
    printf("Erased %d random devices\n", m);
    //print_data_and_coding(k, m, w, psize, data, coding);
    printf("\n");
    
//    Decoding from genuine devices
    start = clock();
    jerasure_bitmatrix_decode(k, m, w, bitmatrix, 0, random, data, coding, w*psize, psize);
    printf("Devices recovered, time elapsed: %.4fs\n", (clock() - (float)start) / CLOCKS_PER_SEC);
    //print_data_and_coding(k, m, w, psize, data, coding);
    printf("\n");
}
