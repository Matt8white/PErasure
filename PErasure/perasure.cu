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

__global__ void smpe(int k, int w, char **dataDevice, char **codingDevice, int psize, int numOfLong)
{
	// TO DO
	int i=0;
	i += 1;
}

int main(int argc, char **argv){
	
	unsigned int m, k, w, i, j, d, r, l, seed, psize;
	unsigned int round;
	int numBytesBDM, numBytesData, numBytesCoding;
	int *matrix, *bitmatrix, *bitmatrixDevice;
	clock_t start;
	char **data, **dataDevice, **dataTemp, **coding, **codingDevice, **codingTemp;
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
    
    //	Allocating GPU memory   
    start = clock();
    numBytesBDM = (m * k) * sizeof(int);
    cudaMalloc(&bitmatrixDevice, numBytesBDM);
	cudaMemcpy(bitmatrixDevice, bitmatrix, numBytesBDM, cudaMemcpyHostToDevice);
    cudaBindTexture(NULL, texture_reference, bitmatrixDevice, numBytesBDM);
    
    numBytesData = k * sizeof(long);
    cudaMalloc(&dataDevice, numBytesData);
    
    numBytesCoding = m * sizeof(long);
    cudaMalloc(&codingDevice, numBytesCoding);
    cudaMemcpy(codingDevice, coding, numBytesCoding, cudaMemcpyHostToDevice);
    
    //	Compute number of rounds
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    round = ceil((float)(psize * w * (k + m)) / free);
    dataTemp = talloc(char *, k);
	for (i = 0; i < k; i++)
		dataTemp[i] = talloc(char, psize/round*w);
	codingTemp = talloc(char *, m);
	for (i = 0; i < m; i++)
		codingTemp[i] = talloc(char, psize/round*w);
    printf("Free mem: %lu\n", free);
    
    for(i=0; i < round; i++){

		// load data chunks
		for(d=0; d < k; d++)
			for(r=0; r < w; r++)
				for(l=0; l < psize / round; l++)
					dataTemp[d][r*psize/round + l] += data[d][r*psize + l + i*psize/round];

		cudaMemcpy(dataDevice, dataTemp, numBytesData, cudaMemcpyHostToDevice);

		for(j=0; j < m; j++)
			smpe<<<dimGrid, dimBlock>>>(k, w, dataDevice, codingTemp+(m*w*psize/round), psize, sizeof(long));
		// copy coding back to main memory
		cudaMemcpy(codingTemp, codingDevice, numBytesCoding, cudaMemcpyDeviceToHost);
		// Extend_Coding_Device(codingTemp, coding, destId);

		cudaFree(dataDevice);
		cudaFree(codingDevice);
	}
    printf("Encoding complete, time elapsed: %.2fs\n", (clock() - (float)start) / CLOCKS_PER_SEC);

    return 0;
}
