#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cuda.h>
#include "jerasure.h"
extern "C"{
	#include "gf_rand.h"
}

using namespace std;

#define talloc(type, num) (type *) malloc(sizeof(type)*(num))

__global__ void smpe()
{
	// TO DO
}

int main(int argc, char **argv){
	
	unsigned int m, k, w, i, j, n, seed, psize;
	unsigned int round;
	int numBytesBDM, numBytesData;
	int *matrix;
	int *bitmatrix;
	int *bitmatrixDevice;
	char **data, **dataDevice;
	dim3 dimBlock(4, 4);
    dim3 dimGrid(4, 4);
    texture<int, 2> texture_reference;
    
    
    if(argc != 5) {
        fprintf(stderr, "Please add arguments m, k, w and size\n");
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
    
    //	Allocating GPU memory   
    
    numBytesBDM = (m * k) * sizeof(int);
    cudaMalloc(&bitmatrixDevice, numBytesBDM);
    cudaMemcpy(bitmatrixDevice, bitmatrix, numBytesBDM, cudaMemcpyHostToDevice);
    cudaBindTexture(NULL, texture_reference, bitmatrixDevice, numBytesBDM)
    
    numBytesData = k * sizeof(long);
    cudaMalloc(&dataDevice, numBytesData);
    cudaMemcpy(dataDevice, data, numBytesData, cudaMemcpyHostToDevice);
    
    numBytesCoding = m * sizeof(long);
    cudaMalloc(&codingDevice, numBytesCoding);
    cudaMemcpy(codingDevice, coding, numBytesCoding, cudaMemcpyHostToDevice);
    
    //	Compute number of rounds
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    round = ceil((psize * w * (k + m)) / free)
    
    for(i=0; i < round; i++){
		// load data chunks
		for(x=0; x < k; x++)
			for(l=0; l < round ; l++)
				dataTemp[x*psize + l] += data[x + l * n]
		for(j=0; j < m; j++)
			smpe<<<dimGrid, dimBlock>>>(k, w, dataDevice, &codingDevice+(i * m), size, sizeof(long));
		// copy coding back to main memory
		cudaMemcpy(coding, codingDevice, numBytesCoding, cudaMemcpyDeviceToHost);
		cudaFree();
	}
    
    return 0;
}
