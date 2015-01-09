#include "TransTable.hpp"
#include <stdio.h>
#include <assert.h>

int main() {
  TransTable tt;
  int bound;
  bool exact;
  std::tie(bound, exact) = tt.get<BLACK>(1, 2);
  assert(bound == N && !exact);
  tt.set(1, 2, 3, true);
  std::tie(bound, exact) = tt.get<BLACK>(1, 2);
  assert(bound == 3 && exact);
}
