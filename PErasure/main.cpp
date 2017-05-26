#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "jerasure.h"

using namespace std;

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
	
	unsigned int m, k, w, i, j, n, seed, size;
	int *matrix;
	int *bitmatrix;
	char **data, **coding;
    srand((unsigned int)time(NULL));
    
    if(argc != 4) {
        fprintf(stderr, "Please add arguments m, k, w\n");
        exit(1);
    }
	if(sscanf(argv[1], "%d", &k) == 0 || k <= 0) {
		fprintf(stderr, "Wrong m\n"); 
		exit(1);
	};
	if (sscanf(argv[2], "%d", &m) == 0 || m <= 0) {
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
    size = 8;
	MOA_Seed(seed);
	data = talloc(char *, k);
	for (i = 0; i < k; i++) {
		data[i] = talloc(char, size);
		MOA_Fill_Random_Region(data[i], size);
	}
    
    coding = talloc(char *, m);
    for (i = 0; i < m; i++) {
        coding[i] = talloc(char, size);
    }
    jerasure_matrix_encode(k, m, w, matrix, data, coding, size);
    
	
//    Printing all the shit
    printf("Encoding Complete:\n");
    printf("\n");
    print_data_and_coding(k, m, w, size, data, coding);
    
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
}
