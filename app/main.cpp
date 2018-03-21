#include <iostream>
#include <stdlib.h>

#include "config.hpp"
#include "boson.hpp"
#include "system.hpp"
#include "wavefunction.hpp"

using namespace std;

/*
 * Simple main program that demontrates how access
 * CMake definitions (here the version number) from source code.
 */
int main() {
    cout << "VMC v" << PROJECT_VERSION_MAJOR << "." << PROJECT_VERSION_MINOR << endl;

    Boson b(3);
    b[0] = 1;
    b[2] = 3;

    cout << b << endl;

    System s(10, 3);

    s[3] = b;

    cout << s << endl;

    return 0;
}
