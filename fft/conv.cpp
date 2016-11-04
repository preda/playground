#include "clutil.h"
#include "time.h"
#include <string.h>
#include <assert.h>
#include <stdlib.h>

#define N 4*1024
#define SIZE (N * N)
#define GS 256

#define K(program, name) Kernel name(program, #name);

int main(int argc, char **argv) {
  time();

  if (argc < 2) {
    printf("Usage: %s <exp> to Lucas-Lehmer test 2^exp - 1 \n", argv[0]);
    exit(1);
  }
  
  int exp = atoi(argv[1]);
  int words = SIZE / 2;
  int bitsPerWord = exp / words + 1;        // 'exp' being prime, 'words' does not divide it.
  if (bitsPerWord < 2) { bitsPerWord = 2; } // Min 2 bits/word.
  int wordsUsed = exp / bitsPerWord + 1;
  
  printf("Lucas-Lehmer test for 2^%d - 1. %d words, %d bits/word, %d words used\n",
         exp, words, bitsPerWord, wordsUsed);
  
  Context c;
  Queue queue(c);
  Program program;
  time("OpenCL init");
  
  program.compileCL2(c, "conv.cl");
  
  K(program, difIniZeropad);
  K(program, difIniZeropadShifted);
  K(program, difStep);
  K(program, dif4Step);
  
  K(program, ditStep);
  K(program, ditFinalShifted);


  K(program, sq4k);
  
  time("Kernels compilation");

  Buf bitsBuf(c, CL_MEM_READ_WRITE /*| CL_MEM_COPY_HOST_PTR*/, sizeof(int) * words, 0);
  int data = 0;
  clEnqueueFillBuffer(queue.queue, bitsBuf.buf, &data, sizeof(data), 0, words, 0, 0, 0);
  data = 4; // LL seed
  queue.writeBlocking(bitsBuf, &data, sizeof(data));

  int *tmp1 = new int[SIZE];

  srandom(100);
  for (int i = 0; i < SIZE; ++i) { tmp1[i] = random(); }
  // for (int i = 0; i < 4 * 1024; ++i) { tmp1[i] = 1; tmp1[4 * 1024 * 1024 + i] = 2; tmp1[8 * 1024 * 1024 + i] = 3; tmp1[12 * 1024 * 1024 + i] = 4; }
  // tmp1[1] = 1;

  Buf buf1(c, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * SIZE, tmp1);
  Buf buf2(c, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * SIZE, tmp1);
  Buf bufTmp(c, CL_MEM_READ_WRITE, sizeof(int) * SIZE, 0);
  time("alloc gpu buffers");

  /*
  sq4k.setArgs(buf1, buf2);
  for (int i = 0; i < 1000; ++i) {
    queue.run(sq4k, GS, words * GS / (64 * 64));
  }
  queue.finish();
  time("sq4k");
  */

  for (int round = 11; round >= 0; round -= 2) {
    difStep.setArgs(round, buf2, bufTmp);
    queue.run(difStep, GS, words);
    difStep.setArgs(round - 1, bufTmp, buf2);
    queue.run(difStep, GS, words);
  }

  for (int round = 5; round >= 0; round -= 2) {
    dif4Step.setArgs(round, buf1, bufTmp);
    queue.run(dif4Step, GS, words / 4);
    dif4Step.setArgs(round - 1, bufTmp, buf1);
    queue.run(dif4Step, GS, words / 4);
  }
    
  queue.readBlocking(buf1, 0, sizeof(int) * SIZE, tmp1);
  int *tmp2 = new int[SIZE];
  queue.readBlocking(buf2, 0, sizeof(int) * SIZE, tmp2);
  int err = 0;
  for (int i = 0; i < SIZE; ++i) {
    if (tmp1[i] != tmp2[i]) {
      ++err;
      if (err > 10) { break; }
      printf("%d %d %d\n", i, tmp1[i], tmp2[i]);
      // i += 256 * 1024;
    }
  }
  time("test");
  
  for (int i = 0; i < 100; ++i) {
    for (int round = 11; round > 0; round -= 2) {
      difStep.setArgs(round, buf2, bufTmp);
      queue.run(difStep, GS, words);
      difStep.setArgs(round - 1, bufTmp, buf2);
      queue.run(difStep, GS, words);
    }
  }
  queue.finish();
  time("dif radix2");
  
  for (int i = 0; i < 100; ++i) {
    for (int round = 5; round > 0; round -= 2) {
      dif4Step.setArgs(round, buf1, bufTmp);
      queue.run(dif4Step, GS, (words * 2) / 8);
      dif4Step.setArgs(round - 1, bufTmp, buf1);
      queue.run(dif4Step, GS, (words * 2) / 8);
    }
  }
  queue.finish();
  time("dif radix4");
  
  /*
  // Initial DIF round on zero-padded input.
  difIniZeropad.setArgs(bitsBuf, buf2);

  for (int i = 0; i < 100; ++i) {
  queue.run(difIniZeropad, GS, SIZE / 4);

  difStep.setArgs(10, buf2, buf1);
  queue.run(difStep, GS, SIZE / 2);
  
  for (int i = 0; i < 5; ++i) {
    difStep.setArgs(9 - i * 2, buf1, buf2);
    queue.run(difStep, GS, SIZE / 2);
    difStep.setArgs(8 - i * 2, buf2, buf1);
    queue.run(difStep, GS, SIZE / 2);
  }
  }
  queue.finish();
  time("dif1");
  */

  
  //difIniZeropadShifted.setArgs(bitsBuf, buf2);
  
  /*
  for (int i = 0; i < 5; ++i) {
    ditStep.setArgs(i * 2, bigBuf, tmpBuf);
    queue.run(ditStep, GS, SIZE / 2);
    ditStep.setArgs(i * 2 + 1, tmpBuf, bigBuf);
    queue.run(ditStep, GS, SIZE / 2);
  }
  queue.finish();
  time("dit");
  
  queue.readBlocking(bigBuf, 0, sizeof(int) * SIZE, big2);
  time("read from gpu");
  */

  /*
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
  */
}
