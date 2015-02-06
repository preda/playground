// (c) Copyright 2014 Mihai Preda. All rights reserved.

#include "Node.hpp"
#include <stdio.h>
#include <assert.h>

template<int C> void doPlay(Node &node, int p) {
  if (node.isSuicide<C>(p)) {
    printf("suicide\n");
    return;
  }
  
  node = node.play<C>(p);
  node.print();
}

int main() {
  Node node;
  node.setup("x....xxxxooo.o..");  
  node.print();
  printf("score %d\n", node.finalScore());
}
