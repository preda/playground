#include "clutil.h"
#include "time.h"
#include <string.h>
#include <assert.h>

#define N 1024
#define WIDTH 1024
#define SIZE (N * WIDTH)
#define GS 256

#define K(program, name) Kernel name(program, #name);

int main(int argc, char **argv) {
  time();
  Context c;
  Queue queue(c);
  Program program;
  time("OpenCL init");
  program.compileCL2(c, "conv.cl");
  K(program, dif);
  K(program, dit);
  time("compile");

  // Buf buf(c, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * 2 * N, d);
  
  int *big1 = new int[SIZE];
  int *big2 = new int[SIZE];
  for (int i = 0; i < SIZE; ++i) {
    big1[i] = (i % 13) + 2;
    big2[i] = (i % 5) - 1;
  }
  
  Buf bigBuf(c, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * SIZE, big1);
  Buf tmpBuf(c, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * SIZE, big2);
  time("gpu buffers");

  for (int i = 0; i < 5; ++i) {
    dif.setArgs(9 - i * 2, bigBuf, tmpBuf);
    queue.run(dif, GS, SIZE / 2);
    dif.setArgs(9 - (i * 2 + 1), tmpBuf, bigBuf);
    queue.run(dif, GS, SIZE / 2);
  }
  queue.finish();
  time("dif");
  
  for (int i = 0; i < 5; ++i) {
    dit.setArgs(i * 2, bigBuf, tmpBuf);
    queue.run(dit, GS, SIZE / 2);
    dit.setArgs(i * 2 + 1, tmpBuf, bigBuf);
    queue.run(dit, GS, SIZE / 2);
  }
  queue.finish();
  time("dit");
  
  queue.readBlocking(bigBuf, 0, sizeof(int) * SIZE, big2);
  time("read from gpu");

  int err = 0;
  for (int i = 0; i < SIZE; ++i) {
    if (big1[i] != big2[i]) {
      printf("%d %d %d\n", i, big1[i], big2[i]);
      ++err;
      if (err > 10) { break; }
    }
  }
  if (!err) {
    printf("OK\n");
  }
  
  delete[] big1;
  delete[] big2;
}
