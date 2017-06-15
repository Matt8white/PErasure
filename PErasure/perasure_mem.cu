#include <stdio.h>
#include <time.h>
#include <math.h>
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

texture<int, 1, cudaReadModeElementType> texBDM;

__global__ void gmpe(int k, int w, int destId, long *dataDevice, long *codingDevice, int numOfLong) {
	
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
		index = destId * k * w * w + i * k * w;
		temp = 0;
		for(dataIdx=0; dataIdx<k; dataIdx++){
			dtPtr = dataDevice + dataIdx * w * numOfLong;
			for(j=0; j<w; j++){
				if(tex1Dfetch(texBDM, index)){
					innerDtPtr = dtPtr + j * numOfLong;
					temp ^= innerDtPtr[longIndex];
				}
				index++;
			}
		}
		codPtr[longIndex] = temp;
	}
}

void extendCodingDevice(long *codingTemp, long *coding, int i, int m, int psize, int offset, int rows){
	int k, j;
	
	for(k=0; k<m; k++)	
		for(j=0; j<rows; j++)
			memcpy((coding + k * psize * rows + psize * j + i * offset), (codingTemp + k * offset * rows + offset * j), sizeof(long) * offset);
}

int main(int argc, char **argv){

	unsigned int m, k, w, i, j, d, r, seed, psize, round;
	int *matrix, *bitmatrix, *bitmatrixDevice;
	long *data, *dataDevice, *dataTemp, *coding, *codingDevice, *codingTemp;
    clock_t start;
    texBDM.filterMode = cudaFilterModePoint;
    texBDM.addressMode[0] = cudaAddressModeClamp;
    
    srand(time(NULL));
    seed = rand();
	MOA_Seed(seed);

    if(argc != 5) {
        fprintf(stderr, "Please add arguments k, m, w and size\n");
        exit(1);
    }
	if(sscanf(argv[1], "%d", &k) == 0 || k <= 0) {
		fprintf(stderr, "Wrong k. It must be strictly postive.\n");
		exit(1);
	}
	if (sscanf(argv[2], "%d", &m) == 0 || m <= 0) {
		fprintf(stderr, "Wrong m. It must be strictly positive.\n");
		exit(1);
	}
	if (sscanf(argv[3], "%d", &w) == 0 || w <= 0 || w > 31) {
		fprintf(stderr, "Wrong w. It must be between 0 and 32.\n");
		exit(1);
	}
	if (sscanf(argv[4], "%d", &psize) == 0 || psize%sizeof(long) != 0){
		fprintf(stderr, "Wrong packetsize. It must be an amount of bytes multiple of long.\n");
		exit(1);
	}
	if((k + m) > (1 << w)) {
		fprintf(stderr, "Wrong combinatio of k, m and w. The following must hold: m + k <= 2^w\n");
		exit(1);
	}
	
	psize = psize/sizeof(long);
	
	int threadPerBlock = min(psize, 1024);
	int nBlocks = ceil((float)psize/threadPerBlock);
	
//    Creating CRS matrix and BDM

	matrix = talloc(int, m*k);
	for (i = 0; i < m; i++) {
		for (j = 0; j < k; j++) {
			matrix[i*k+j] = galois_single_divide(1, i ^ (m + j), w);
		}
	}

	bitmatrix = jerasure_matrix_to_bitmatrix(k, m, w, matrix);

//	Generating fake random data

	data = talloc(long , k*w*psize);
	for (i = 0; i < k; i++) {
		for(j=0; j< w*psize; j++)
			*(data + i*psize*w + j) = 97 + rand()%26;
	}
	
//	Allocating space for coding devices

	coding = talloc(long , m * w * psize);
	
//	Allocating GPU memory
    
    cudaMalloc(&bitmatrixDevice, m*k*w*w*sizeof(int));
    cudaMemcpy(bitmatrixDevice, bitmatrix, m*k*w*w*sizeof(int), cudaMemcpyHostToDevice);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
    cudaBindTexture(0, texBDM, bitmatrixDevice, channelDesc, m*k*w*w*sizeof(int));
    
//	Computing number of rounds
    
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    
    round = ceil((float)(psize * w * (k + m) * sizeof(long)) / free);

    dataTemp = talloc(long , k * w * (psize/round));
	
	codingTemp = talloc(long, m * w * (psize/round));

	start = clock();
    for(i = 0; i < round; i++){
		
		// load data chunks when needed
		if(round > 1){
			for(d = 0; d < k; d++)
				for(r = 0; r < w; r++)
					memcpy((dataTemp + d * w *(psize/round) + r * (psize/round)), (data + d * w * psize + i * psize/round + r * psize), sizeof(long) * (psize/round));
											
			cudaMalloc(&dataDevice, k * w * (psize/round) * sizeof(long));
			cudaMalloc(&codingDevice, m * w * (psize/round) * sizeof(long));
			cudaMemcpy(dataDevice, dataTemp, k * w * (psize/round) * sizeof(long), cudaMemcpyHostToDevice);
			cudaMemcpy(codingDevice, coding, m * w * (psize/round) * sizeof(long), cudaMemcpyHostToDevice);
		} //else load all the data
		else{
			cudaMalloc(&dataDevice, k * w * (psize/round) * sizeof(long));
			cudaMalloc(&codingDevice, m * w * (psize/round) * sizeof(long));
			cudaMemcpy(dataDevice, data, k * w * (psize/round) * sizeof(long), cudaMemcpyHostToDevice);
			cudaMemcpy(codingDevice, codingTemp, m * w * (psize/round) * sizeof(long), cudaMemcpyHostToDevice);
		}
		
		for(j = 0; j < m; j++)
			gmpe<<<nBlocks, threadPerBlock>>>(k, w, j, dataDevice, codingDevice, (psize/round));
			
		// copy coding back to main memory
		cudaDeviceSynchronize();
		cudaMemcpy(codingTemp, codingDevice, m * w * (psize/round) * sizeof(long), cudaMemcpyDeviceToHost);
		extendCodingDevice(codingTemp, coding, i, m, psize, (psize/round), w);

		cudaFree(dataDevice);
		cudaFree(codingDevice);
	}
    printf("Encoding complete, time elapsed: %.8fs\n", (clock() - (float)start) / CLOCKS_PER_SEC);

    cudaUnbindTexture(texBDM);

    return 0;
}
