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
	
	unsigned int m, k, w, i, j, n, seed, psize;
	int *matrix;
	int *bitmatrix;
	char **data, **coding;
    srand((unsigned int)time(NULL));
    
    if(argc != 5) {
        fprintf(stderr, "Please add arguments m, k, w and packetsize\n");
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
    
    printf("Encoding Complete\n");
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
    printf("\n");
    
//    Decoding from genuine devices
    jerasure_bitmatrix_decode(k, m, w, matrix, 0, random, data, coding, w*psize, psize);
    printf("Devices recovered\n");
    printf("\n");
}
