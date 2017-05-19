//
//  main.cpp
//  PErasure
//
//  Created by Mattia Bianchi on 16/05/17.
//  Copyright © 2017 BianchiDagrada. All rights reserved.
//

#include <iostream>
#include <omp.h>
#include <string>
using namespace std;

int main(int argc, const char * argv[]) {
    // insert code here...
    #pragma omp parallel num_threads(3)
    {
        std::cout << "Hello, World!\n";
    }
    std::string s = std::to_string(omp_get_num_devices());
    std::cout << s + "\n";
    return 0;
}
