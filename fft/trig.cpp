// #include <stdio.h>
// #include <math.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

void generate(int endLine, int endCol) {
  using namespace std;
  for (int line = 1; line < endLine; ++line) {
    for (int col = 0; col < endCol; ++col) {
      auto angle = - M_PIl * line * col / (endLine * endCol / 2);
      cout << setw(22) << cosl(angle) << ", "
           << setw(22) << sinl(angle) << ", ";
      if ((col & 1) == 1) { cout << endl; }
    }
  }

}

int main() {
  using namespace std;
  cout.precision(numeric_limits<double>::max_digits10);
  generate(16, 16);
  generate(2, 16);
  generate(16, 256);
}
