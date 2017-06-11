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

int main(int argc, char **argv){
	
	unsigned int m, k, w, i, j, seed, psize;
	int *matrix, *bitmatrix;
	char **data, **coding;
	clock_t start;
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
	printf("Encoding complete, time elapsed: %.8fs\n", (clock() - (float)start) / CLOCKS_PER_SEC);
	return 0;
}
