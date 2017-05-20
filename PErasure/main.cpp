//
//  main.cpp
//  PErasure
//
//  Created by Mattia Bianchi on 16/05/17.
//  Copyright © 2017 BianchiDagrada. All rights reserved.
//

#include <iostream>
#include <omp.h>
#include <openacc.h>
#include <string>
using namespace std;

int main(int argc, const char * argv[]) {
    // insert code here...
    #pragma omp parallel num_threads(3)
    {
        std::cout << "Hello, World!\n";
    }
    std::string s = std::to_string(acc_get_device_type());
    std::cout << s + "\n";
    return 0;
//    int n = 10240; float a = 2.0f; float b = 3.0f;
//    float *x = (float*) malloc(n * sizeof(float));
//    float *y = (float*) malloc(n * sizeof(float));
//    double start = omp_get_wtime();
//    #pragma omp target data map(to:x)
//    {
//        #pragma omp target map(tofrom:y)
//        #pragma omp teams
//        #pragma omp distribute parallel for
//        for (int i = 0; i < n; ++i){
//            y[i] = a*x[i] + y[i];
//        }
//        #pragma omp target map(tofrom:y)
//        for (int i = 0; i < n; ++i){
//              y[i] = b*x[i] + y[i];
//        }
//        
//    }
//    std::cout << "Time: " << (omp_get_wtime() - start) * 1000.0 << " ms" <<std::endl;
//    free(x);
//    free(y);
//    return 0;
}

//#include <stdio.h>
//#include <stdlib.h>
//
//int main(int argc, char* argv[])
//{
//    
//    
//    int n = 1024;
//    
//    double* x = (double*)malloc(sizeof(double) * n);
//    double* y = (double*)malloc(sizeof(double) * n);
//    
//    double idrandmax = 1.0 / RAND_MAX;
//    double a = idrandmax * rand();
//    for (int i = 0; i < n; i++)
//    {
//        x[i] = idrandmax * rand();
//        y[i] = idrandmax * rand();
//    }
//    
//#pragma omp target data map(tofrom: x[0:n],y[0:n])
//    {
//#pragma omp target
//#pragma omp for
//        for (int i = 0; i < n; i++)
//            y[i] += a * x[i];
//    }
//    
//    double avg = 0.0, min = y[0], max = y[0];
//    for (int i = 0; i < n; i++)
//    {
//        avg += y[i];
//        if (y[i] > max) max = y[i];
//        if (y[i] < min) min = y[i];
//    }
//    
//    printf("min = %f, max = %f, avg = %f\n", min, max, avg / n);
//    
//    free(x);
//    free(y);
//    
//    return 0;
//}
