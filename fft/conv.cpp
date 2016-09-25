#include "clutil.h"
#include "time.h"
#include <string.h>
#include <assert.h>

#define N 2048
#define GS 256

#define K(program, name) Kernel name(program, #name);

/*
void print(double *d, int step = 1) {
  for (int i = 0; i < N; i += step) {
    printf("%3d: %f %f\n", i, d[i*2], d[i*2 + 1]);
  }
}
*/

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
  
  int *big1 = new int[1024 * 2048];
  int *big2 = new int[1024 * 2048];
  for (int i = 0; i < 1024 * 2048; ++i) {
    big1[i] = (i % 13) + 2;
    big2[i] = (i % 5) - 1;
  }
  
  Buf bigBuf1(c, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * 1024 * 2048, big1);
  Buf bigBuf2(c, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * 1024 * 2048, big2);
  time("bigBuf");

  for (int rep = 0; rep < 1; ++rep) {
  for (int round = 9; round >= 0; --round) {
    if (round & 1) {
      dif.setArgs(bigBuf1, bigBuf2, round);
    } else {
      dif.setArgs(bigBuf2, bigBuf1, round);
    }
    queue.run(dif, 256, 256 * 512 * 8);
  }
  for (int round = 0; round < 10; ++round) {
    if (round & 1) {
      dit.setArgs(bigBuf2, bigBuf1, round);
    } else {
      dit.setArgs(bigBuf1, bigBuf2, round);
    }
    queue.run(dit, 256, 256 * 512 * 8);
  }
  }
  queue.readBlocking(bigBuf1, 0, sizeof(int) * 1024 * 2048, big2);
  time("fft");

  int err = 0;
  for (int i = 0; i < 1024 * 2048; ++i) {
    if (big1[i] != big2[i] / 1024) {
      printf("%d %d %d\n", i, big1[i], big2[i]);
      ++err;
      if (err > 10) { break; }
    }
  }

  

  /*
  for (int i = 0; i < 2; ++i) {
    conv.setArgs(bigBuf2, i);
    queue.run(conv, 256, 512 * 256);
  }
  queue.readBlocking(bigBuf2, 0, sizeof(int) * 1024 * 2048, big2);
  
  for (int i = 0; i < 1; ++i) {
    if (i & 1) {
      dif.setArgs(bigBuf2, bigBuf1, i);
    } else {
      dif.setArgs(bigBuf1, bigBuf2, i);
    }
    queue.run(dif, 256, 256 * 256 * 4);
              // 512 * 2048);
  }
  
  queue.readBlocking(bigBuf2, 0, sizeof(int) * 1024 * 2048, big1);
  */
  
  /*  
  int round = 0;
  conv.setArgs(bigBuf1, round);
  dif.setArgs(bigBuf1, bigBuf2, round);
  // queue.run(conv, GS, GS * 512);
  queue.run(dif, 256, 512 * 2048);
  queue.finish();
  time("warm up");
  
  for (int i = 0; i < 100; ++i) {
    for (int round = 0; round < 10; ++round) {
      // conv.setArg(1, round);
      if (round & 1) {
        dif.setArgs(bigBuf2, bigBuf1, round);
      } else {
        dif.setArgs(bigBuf1, bigBuf2, round);
      }
      queue.run(dif, 256, 512 * 2048);
    }
  }
  
  queue.finish();
  time("dif");
  // queue.readBlocking(bigBuf, 0, sizeof(double), big);
  */
  
  delete[] big1;
  delete[] big2;
}
