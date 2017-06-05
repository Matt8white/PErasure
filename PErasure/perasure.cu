#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cuda.h>
#include "jerasure.h"
#include <algorithm>
extern "C"{
	#include "gf_rand.h"
}

using namespace std;

#define talloc(type, num) (type *) malloc(sizeof(type)*(num))

//__global__ void smpe(int k, int w, int *bitmatrixDevice, int destId, char *dataDevice, char *codingDevice, int dataSize, int numOfLong) {
	//__shared__ char sharedData[dataSize];
	//int blockNumInGrid, threadsPerBlock, threadNumInBlock, tId;
	//blockNumInGrid = blockIdx.x + gridDim.x * blockIdx.y;
	//threadsPerBlock = blockDim.x * blockDim.y;
	//threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
	//tId = blockNumInGrid * threadsPerBlock + threadNumInBlock;
	
	//int rowIdx = tId / numOfLong;
	//int colIdx =  tId % numOfLong;
	//int temp = 0;
	//int wordsPerThread = max(1, (dataSize/sizeof(long))/(gridDim.x * gridDim.y * blockDim.x * blockDim.y));
	
	//if(tId * sizeof(long) >= dataSize)
		//return;
	
	//for(dataIdx = 0; dataIdx < k; dataIdx++)
		//memcpy((char *)&sharedData, (char *)(dataDevice + dataIdx * dataSize + tId * ), sizeof(long)); //capire bene cosa succede qui
		////sharedData = *(dataDevice + dataIdx * psize * w)
		//__syncthreads();
		//index = 0;
		//for(i=0; i<w; i++) //qui manca qualcosa 
			//sdIndex = dataIdx + i * psize + colIndex;
			//temp ^= (*(bitmatrixDevice + index) & sharedData[sdIndex]);
			//index++;
		//__syncthreads();
	//codingDevice = 
	
//}

__global__ void gmpe(int k, int w, int *bitmatrixDevice, int destId, long *dataDevice, long *codingDevice, int dataSize, int numOfLong) {
	
	int blockNumInGrid, threadsPerBlock, threadNumInBlock, tId;
	blockNumInGrid = blockIdx.x + gridDim.x * blockIdx.y;
	threadsPerBlock = blockDim.x * blockDim.y;
	threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
	tId = blockNumInGrid * threadsPerBlock + threadNumInBlock;
	
	int longIndex = tId % numOfLong;
	int index, dataIdx, i, j;
	long temp;
	long *codPtr, *dtPtr, *innerDtPtr;
	
	if( tId >= numOfLong)
		return;
	
	for(i=0; i<w; i++){
		codPtr = codingDevice + destId * w * numOfLong + i * numOfLong;
		index = destId * k * w * w + i * w;
		temp = 0;
		for(dataIdx=0; dataIdx<k; dataIdx++){
			dtPtr = dataDevice + dataIdx * w * numOfLong;
			for(j=0; j<w; j++){
				if(bitmatrixDevice[index]){
					innerDtPtr = dtPtr + j * numOfLong;
					temp ^= innerDtPtr[longIndex];
				}
				index++;
			}
		}
		codPtr[longIndex] = temp;
	}
}

int main(int argc, char **argv){

	unsigned int m, k, w, i, j, d, r, l, seed, psize;
	unsigned int round;
	//int numBytesBDM, numBytesData, numBytesCoding;
	int *matrix, *bitmatrix, *bitmatrixDevice;
	clock_t start;
	long *data, *dataDevice, *dataTemp, *coding, *codingDevice, *codingTemp;
	dim3 dimGrid(1, 1);
	dim3 dimBlock(16, 1);
    texture<int, 2> texture_reference;
    
    srand(time(NULL));


    if(argc != 5) {
        fprintf(stderr, "Please add arguments k, m, w and size\n");
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
	psize = psize/sizeof(long);
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
	//jerasure_print_bitmatrix(bitmatrix, m*w, k*w, w);

//    Generating fake random data
	data = talloc(long , k*w*psize);
	for (i = 0; i < k; i++) {
		//MOA_Fill_Random_Region(data+i*psize*w, psize*w);
		for(j=0; j< w*psize; j++)
			*(data + i*psize*w + j) = 97 + rand()%26;
	}

	coding = talloc(long , m * w * psize);
	
    //	Allocating GPU memory
    start = clock();
    
    cudaMalloc(&bitmatrixDevice, m*k*w*w*sizeof(int));
    cudaMemcpy(bitmatrixDevice, bitmatrix, m*k*w*w*sizeof(int), cudaMemcpyHostToDevice);
    
    //	Compute number of rounds
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    
    round = ceil((float)(psize * w * (k + m)) / free);

    printf("Free mem: %lu\n", free);
    
	dataTemp = talloc(long , k * w * (psize/round));
	cudaMalloc(&dataDevice, k * w * (psize/round) * sizeof(long));
	
	codingTemp = talloc(long, m * w * (psize/round));
	cudaMalloc(&codingDevice, m * w * (psize/round) * sizeof(long));

    for(i = 0; i < round; i++){
	
		// load data chunks

		for(d = 0; d < k; d++)
			for(r = 0; r < w; r++)
				for(l = 0; l < psize / round; l++)
					*(dataTemp + d * w *(psize/round) + r * (psize/round) + l) = *(data + d * w * psize + i * psize/round + r * psize + l);

		cudaMemcpy(dataDevice, dataTemp, k * w * (psize/round) * sizeof(long), cudaMemcpyHostToDevice);
		cudaMemcpy(codingDevice, codingTemp, m * w * (psize/round) * sizeof(long), cudaMemcpyHostToDevice);

		for(j = 0; j < m; j++)
			//smpe<<<dimGrid, dimBlock>>>(k, w, bitmatrixDevice + j * w * w * k, j, dataDevice, codingDevice, (psize/round) * w, sizeof(long));
			gmpe<<<dimGrid, dimBlock>>>(k, w, bitmatrixDevice + j * w * w * k, j, dataDevice, codingDevice, (psize/round) * w, (psize/round));
		// copy coding back to main memory
		
		cudaMemcpy(codingTemp, codingDevice, m * w * (psize/round) * sizeof(long), cudaMemcpyDeviceToHost);
		// Extend_Coding_Device(codingTemp, coding, destId);
		

		cudaFree(dataDevice);
		cudaFree(codingDevice);
	}
    printf("Encoding complete, time elapsed: %.2fs\n", (clock() - (float)start) / CLOCKS_PER_SEC);
    
    for(i = 0; i < k; i++){
		for(j = 0; j < w * psize; j++)
			printf("%ld ", *(dataTemp+i*w*psize + j));
		printf("\n");
	}
	printf("\n");
	
	for(i = 0; i < m; i++){
		for(j = 0; j < w * psize; j++)
			printf("%ld ", *(codingTemp + i*w*psize + j));
		printf("\n");
	}
	printf("\n");

    return 0;
}
