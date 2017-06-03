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

__global__ void smpe(int k, int w, int *bitmatrixDevice, char *dataDevice, char *codingDevice, int psize, int numOfLong) {
	__shared__ long sharedData[psize * w];
	int blockNumInGrid, threadsPerBlock, threadNumInBlock, tId;
	blockNumInGrid = blockIdx.x + gridDim.x * blockIdx.y;
	threadsPerBlock = blockDim.x * blockDim.y;
	threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
	tId = blockNumInGrid * threadsPerBlock + threadNumInBlock;
	
	int rowIdx = tId / numOfLong;
	int colIdx =  tId % numOfLong;
	
	
	//int tidx = blockIdx.x*blockDim.x + threadIdx.x;
	//int tidy = blockIdx.y*blockDim.y + threadIdx.y;
	//int lenData = psize;
	//int numOfData = ceil((float)(lenData) / (blockDim.x * blockDim.y));
	//int base = numOfData * (tidx + tidy);

	//int i, j;
	//if(base < lenData){
		//for(i=0; i < w; i++)
			//for(j=0; j < numOfData; j++)
				//*(codingDevice + base + i*lenData + j) = 'a';	
	//}
	
	if(tId * sizeof(long) > psize)
		return;
	
	for(dataIdx = 0; dataIdx < k; dataIdx++)
		memcpy((char *)&sharedData, (char *)(dataDevice + dataIdx * psize * w), psize * w * sizeof(long)); //capire bene cosa succede qui
		//sharedData = *(dataDevice + dataIdx * psize * w)
		__syncthreads();
		index = 0;
		for(i=0; i<w; i++) //qui manca qualcosa 
			sdIndex = dataIdx + i * psize + colIndex;
			temp ^= (*(bitmatrixDevice + index) & sharedData[sdIndex]);
			index++;
		__syncthreads();
	codingDevice = 
				
		
		
		
	
}

int main(int argc, char **argv){

	unsigned int m, k, w, i, j, d, r, l, seed, psize;
	unsigned int round;
	//int numBytesBDM, numBytesData, numBytesCoding;
	int *matrix, *bitmatrix, *bitmatrixDevice;
	clock_t start;
	char *data, *dataDevice, *dataTemp, *coding, *codingDevice, *codingTemp;
	dim3 dimBlock(4, 4);
    dim3 dimGrid(4, 4);
    texture<int, 2> texture_reference;
    
    srand(time(NULL));


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
	data = talloc(char , k*w*psize);
	for (i = 0; i < k; i++) {
		//MOA_Fill_Random_Region(data+i*psize*w, psize*w);
		for(j=0; j< w*psize; j++)
			*(data + i*psize*w + j) = 97 + rand()%26;
	}

	coding = talloc(char , m * w * psize);
	
	//for(i = 0; i < k; i++){
		//for(j = 0; j < w * psize; j++)
			//printf("%c ", *(data+i*w*psize + j));
		//printf("\n");
	//}
	//printf("\n");
	
	//for(i = 0; i < k; i++){
		//for(j = 0; j < w * psize /2; j++)
			//printf("%c ", *(dataTemp + i*w*psize/2 + j));
		//printf("\n");
	//}
	//printf("\n");
	
	
    //	Allocating GPU memory
    start = clock();
    
    cudaMalloc(&bitmatrixDevice, m*k*w*w*sizeof(int));
    cudaMemcpy(bitmatrixDevice, bitmatrix, m*k*w*w*sizeof(int), cudaMemcpyHostToDevice);
    
    //	Compute number of rounds
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    
    round = ceil((float)(psize * w * (k + m)) / free);

    printf("Free mem: %lu\n", free);
    
	dataTemp = talloc(char , k * w * (psize/round));
	cudaMalloc(&dataDevice, k * w * (psize/round));
	
	codingTemp = talloc(char, m * w * (psize/round));
	cudaMalloc(&codingDevice, m * w * (psize/round));

    for(i = 0; i < round; i++){
	
		// load data chunks

		for(d = 0; d < k; d++)
			for(r = 0; r < w; r++)
				for(l = 0; l < psize / round; l++)
					*(dataTemp + d * w *(psize/round) + r * (psize/round) + l) = *(data + d * w * psize + i * psize/round + r * psize + l);

		cudaMemcpy(dataDevice, dataTemp, k * w * (psize/round), cudaMemcpyHostToDevice);
		cudaMemcpy(codingDevice, codingTemp, m * w * (psize/round), cudaMemcpyHostToDevice);

		for(j = 0; j < m; j++)
			smpe<<<dimGrid, dimBlock>>>(k, w, bitmatrixDevice + j * w * w * k, dataDevice, codingDevice + j * w * (psize/round), (psize/round), sizeof(long));
		// copy coding back to main memory
		cudaMemcpy(codingTemp, codingDevice, m * w * (psize/round), cudaMemcpyDeviceToHost);
		// Extend_Coding_Device(codingTemp, coding, destId);

		cudaFree(dataDevice);
		cudaFree(codingDevice);
	}
    printf("Encoding complete, time elapsed: %.2fs\n", (clock() - (float)start) / CLOCKS_PER_SEC);
    
    for(i = 0; i < k; i++){
		for(j = 0; j < w * psize; j++)
			printf("%c ", *(dataTemp+i*w*psize + j));
		printf("\n");
	}
	printf("\n");
	
	for(i = 0; i < m; i++){
		for(j = 0; j < w * psize; j++)
			printf("%c ", *(codingTemp + i*w*psize + j));
		printf("\n");
	}
	printf("\n");

    return 0;
}
