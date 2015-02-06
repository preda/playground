#include "Node.hpp"
#include <stdio.h>
#include <assert.h>

int main() {
  Node n;
  n.setup("\
.xx.\
xx.o\
xx.o\
xxxx\
");
  n.print();
}
