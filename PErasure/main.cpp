#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "jerasure.h"

#define talloc(type, num) (type *) malloc(sizeof(type)*(num))

static uint32_t MOA_X[5];

uint32_t MOA_Random_32() {
  uint64_t sum;
  sum = (uint64_t)2111111111UL * (uint64_t)MOA_X[3] +
     (uint64_t)1492 * (uint64_t)(MOA_X[2]) +
     (uint64_t)1776 * (uint64_t)(MOA_X[1]) +
     (uint64_t)5115 * (uint64_t)(MOA_X[0]) +
     (uint64_t)MOA_X[4];
  MOA_X[3] = MOA_X[2];  MOA_X[2] = MOA_X[1];  MOA_X[1] = MOA_X[0];
  MOA_X[4] = (uint32_t)(sum >> 32);
  MOA_X[0] = (uint32_t)sum;
  return MOA_X[0];
}

uint64_t MOA_Random_64() {
  uint64_t sum;

  sum = MOA_Random_32();
  sum <<= 32;
  sum |= MOA_Random_32();
  return sum;
}

void MOA_Random_128(uint64_t *x) {
  x[0] = MOA_Random_64();
  x[1] = MOA_Random_64();
  return;
}

uint32_t MOA_Random_W(int w, int zero_ok)
{
  uint32_t b;

  do {
    b = MOA_Random_32();
    if (w == 31) b &= 0x7fffffff;
    if (w < 31)  b %= (1 << w);
  } while (!zero_ok && b == 0);
  return b;
}

void MOA_Seed(uint32_t seed) {
  int i;
  uint32_t s = seed;
  for (i = 0; i < 5; i++) {
    s = s * 29943829 - 1;
    MOA_X[i] = s;
  }
  for (i=0; i<19; i++) MOA_Random_32();
}


void MOA_Fill_Random_Region (void *reg, int size)
{
  uint32_t *r32;
  uint8_t *r8;
  int i;

  r32 = (uint32_t *) reg;
  r8 = (uint8_t *) reg;
  for (i = 0; i < size/4; i++) r32[i] = MOA_Random_32();
  for (i *= 4; i < size; i++) r8[i] = MOA_Random_W(8, 1);
}


int main(int argc, char **argv){
	
	unsigned int m, k, w, i, j, n, seed, size;
	int *matrix;
	int *bitmatrix;
	char **data;
	
    
    if(argc != 4) {
        fprintf(stderr, "Please add arguments m, k, w\n");
        exit(1);
    }
	if(sscanf(argv[1], "%d", &m) == 0 || m <= 0) {
		fprintf(stderr, "Wrong m\n"); 
		exit(1);
	};
	if (sscanf(argv[2], "%d", &k) == 0 || k <= 0) {
		fprintf(stderr, "Wrong k\n"); 
		exit(1);
	}; 
	if (sscanf(argv[3], "%d", &w) == 0 || w <= 0 || w > 31) {
		fprintf(stderr, "Wrong w\n"); 
		exit(1);
	}; 
	if((k + m) > (1 << w)) {
		fprintf(stderr, "Wrong w, the following must hold: m + k <= 2^w\n"); 
		exit(1);
	}; 

	matrix = talloc(int, m*k);

	for (i = 0; i < m; i++) {
		for (j = 0; j < k; j++) {
			n = i ^ ((1 << w) - 1 - j);
			matrix[i*k+j] = (n == 0) ? 0 : galois_single_divide(1, n, w);
		}
	}
	seed = 42;
	size = 3;
	MOA_Seed(seed);
	data = talloc(char *, k);
	for (i = 0; i < k; i++) {
		data[i] = talloc(char, size);
		MOA_Fill_Random_Region(data[i], size);
	}
	
	
	
	bitmatrix = jerasure_matrix_to_bitmatrix(m, k, w, matrix);
	jerasure_print_matrix(matrix, m, k, w);
	jerasure_print_bitmatrix(bitmatrix, m*w, k*w, w);
}
