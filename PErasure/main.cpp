#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "jerasure.h"

#define talloc(type, num) (type *) malloc(sizeof(type)*(num))

int main(int argc, char **argv){
	
	unsigned int m, k, w, i, j, n;
	int *matrix;
	int *bitmatrix;

	if(sscanf(argv[1], "%d", &m) == 0 || k <= 0){
		fprintf(stderr, "Wrong m\n"); 
		exit(1);
	};
	if (sscanf(argv[2], "%d", &k) == 0 || m <= 0){
		fprintf(stderr, "Wrong k\n"); 
		exit(1);
	}; 
	if (sscanf(argv[3], "%d", &w) == 0 || w <= 0 || w > 31){
		fprintf(stderr, "Wrong w\n"); 
		exit(1);
	}; 
	if((k + m) > (1 << w)){
		fprintf(stderr, "Wrong w, the following must hold: m + k <= 2^w\n"); 
		exit(1);
	}; 

	matrix = talloc(int, m*k);

	for (i = 0; i < m; i++) {
		for (j = 0; j < k; j++) {
			n = i ^ ((1 << w)-1-j);
			matrix[i*k+j] = (n == 0) ? 0 : galois_single_divide(1, n, w);
		}
	}
	
	bitmatrix = jerasure_matrix_to_bitmatrix(m, k, w, matrix);
	jerasure_print_matrix(matrix, m, k, w);
	jerasure_print_bitmatrix(bitmatrix, m*w, k*w, w);
}
