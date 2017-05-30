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
	
	unsigned int m, k, w, i, j, n, seed, pzise;
	int *matrix;
	int *bitmatrix;
	char **data, **coding;
    srand((unsigned int)time(NULL));
    
    if(argc != 6) {
        fprintf(stderr, "Please add arguments m, k, w\n");
        exit(1);
    }
	if(sscanf(argv[1], "%d", &k) == 0 || k <= 0) {
		fprintf(stderr, "Wrong m\n"); 
		exit(1);
	}
	if (sscanf(argv[2], "%d", &m) == 0 || m <= 0) {
		fprintf(stderr, "Wrong k\n"); 
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
	matrix = talloc(int, m*k);
	for (i = 0; i < m; i++) {
		for (j = 0; j < k; j++) {
			n = i ^ ((1 << w) - 1 - j);
			matrix[i*k+j] = (n == 0) ? 0 : galois_single_divide(1, n, w);
		}
	}
    bitmatrix = jerasure_matrix_to_bitmatrix(m, k, w, matrix);

//    Generating fake random data
	seed = rand();
	MOA_Seed(seed);
	data = talloc(char *, k);
	for (i = 0; i < k; i++) {
		data[i] = talloc(char, psize*w);
		MOA_Fill_Random_Region(data[i], psize*w);
	}

	coding = talloc(char *, m);
	for (i = 0; i < m; i++) {
		coding[i] = talloc(char, psize*w);
	}
    jerasure_bitmatrix_encode(k, m, w, bitmatrix, data, coding, w*psize, psize);
    

    printf("Encoding Complete:\n");
    printf("\n");
    print_data_and_coding(k, m, w, psize, data, coding);
    
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
            bzero(data[random[i]], size);
        else bzero(coding[random[i] - k], size);
    }
    printf("Erased %d random devices:\n", m);
    printf("\n");
    print_data_and_coding(k, m, w, size, data, coding);
    
//    Decoding from genuine devices
    jerasure_matrix_decode(k, m, w, matrix, 0, random, data, coding, size);
    printf("State of the system after decoding:\n");
    printf("\n");
    print_data_and_coding(k, m, w, size, data, coding);
    
    int *free, *total;
    cudaMemGetInfo(free, total);
    
    printf("Free: %d \t Total: %d\n", free, total);
}
