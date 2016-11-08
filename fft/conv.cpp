#include "clutil.h"
#include "time.h"
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <memory>

#define N 4*1024
#define SIZE (N * N)
#define GS 256

#define K(program, name) Kernel name(program, #name);

bool checkEqual(Queue *queue, Buf *buf1, Buf *buf2, int size) {
  std::unique_ptr<int[]> tmp1(new int[size]);
  std::unique_ptr<int[]> tmp2(new int[size]);
  queue->readBlocking(buf1, 0, sizeof(int) * size, tmp1.get());
  queue->readBlocking(buf2, 0, sizeof(int) * size, tmp2.get());
  int err = 0;
  for (int i = 0; i < size; ++i) {
    if (tmp1[i] != tmp2[i]) {
      printf("%d %d %d\n", i, tmp1[i], tmp2[i]);
      if (++err >= 10) { return false; }
    }
  }
  return true;
}

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
  
  K(program, dif2);
  K(program, dif4);
  K(program, dif8);
  
  K(program, dit2);
  K(program, dit4);
  K(program, dit8);

  K(program, sq4k);
  
  time("Kernels compilation");

  /*
  Buf bitsBuf(c, CL_MEM_READ_WRITE, sizeof(int) * words, 0);
  int data = 0;
  clEnqueueFillBuffer(queue.queue, bitsBuf.buf, &data, sizeof(data), 0, words, 0, 0, 0);
  data = 4; // LL seed
  queue.writeBlocking(bitsBuf, &data, sizeof(data));
  */

  int *data = new int[SIZE];
  
  srandom(0);
  for (int i = 0; i < SIZE; ++i) { data[i] = (random() & 0xffffff) - (1 << 23); }
  time("random");
  
  Buf buf1(c, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * SIZE, data);
  Buf buf2(c, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * SIZE, data);
  Buf bufTmp(c, CL_MEM_READ_WRITE, sizeof(int) * SIZE, 0);
  time("alloc gpu buffers");

  for (int round = 3; round >= 0; round -= 2) {
    dif8.setArgs(round, buf1, bufTmp);
    queue.run(dif8, GS, SIZE / 32);
    dif8.setArgs(round - 1, bufTmp, buf1);
    queue.run(dif8, GS, SIZE / 32);
  }
  
  std::unique_ptr<long[]> tmpLong1(new long[SIZE]);
  {
    std::unique_ptr<int[]> tmp1(new int[SIZE]);
    queue.readBlocking(&buf1, 0, sizeof(int) * SIZE, tmp1.get());
    for (int i = 0; i < SIZE; ++i) {
      tmpLong1[i] = tmp1[i];
    }
  }
  Buf bufLong1(c, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(long) * SIZE, tmpLong1.get());
  Buf bufLongTmp(c, CL_MEM_READ_WRITE, sizeof(long) * SIZE, 0);

  for (int round = 0; round < 4; round += 2) {
    dit8.setArgs(round, bufLong1, bufLongTmp);
    queue.run(dit8, GS, SIZE / 32);
    dit8.setArgs(round + 1, bufLongTmp, bufLong1);
    queue.run(dit8, GS, SIZE / 32);
  }

  queue.readBlocking(&bufLong1, 0, sizeof(long) * SIZE, tmpLong1.get());
  int err = 0;
  for (int i = 0; i < SIZE; ++i) {
    if (data[i] != tmpLong1[i]) {
      printf("%d %d %ld\n", i, data[i], tmpLong1[i]);
      if (++err >= 10) { exit(1); }
    }
  }  
  time("OK FFT radix8 round-trip");
      
  for (int i = 0; i < 100; ++i) {
    for (int round = 11; round > 0; round -= 2) {
      dif2.setArgs(round, buf2, bufTmp);
      queue.run(dif2, GS, words);
      dif2.setArgs(round - 1, bufTmp, buf2);
      queue.run(dif2, GS, words);
    }
  }
  queue.finish();
  time("perf DIF2");
  
  for (int i = 0; i < 100; ++i) {
    for (int round = 5; round > 0; round -= 2) {
      dif4.setArgs(round, buf1, bufTmp);
      queue.run(dif4, GS, (words * 2) / 8);
      dif4.setArgs(round - 1, bufTmp, buf1);
      queue.run(dif4, GS, (words * 2) / 8);
    }
  }
  queue.finish();
  time("perf DIF4");

  for (int i = 0; i < 100; ++i) {
    for (int round = 3; round > 0; round -= 2) {
      dif8.setArgs(round, buf1, bufTmp);
      queue.run(dif8, GS, SIZE / 32);
      dif8.setArgs(round - 1, bufTmp, buf1);
      queue.run(dif8, GS, SIZE / 32);
    }
  }
  queue.finish();
  time("perf DIF8");

  for (int i = 0; i < 100; ++i) {
    for (int round = 0; round < 12; round += 2) {
      dit2.setArgs(round, bufLong1, bufLongTmp);
      queue.run(dit2, GS, SIZE / 2);
      dit2.setArgs(round + 1, bufLongTmp, bufLong1);
      queue.run(dit2, GS, SIZE / 2);
    }
  }
  queue.finish();
  time("perf DIT2");

  for (int i = 0; i < 100; ++i) {
    for (int round = 0; round < 6; round += 2) {
      dit4.setArgs(round, bufLong1, bufLongTmp);
      queue.run(dit4, GS, SIZE / 8);
      dit4.setArgs(round + 1, bufLongTmp, bufLong1);
      queue.run(dit4, GS, SIZE / 8);
    }
  }
  queue.finish();
  time("perf DIT4");

  for (int i = 0; i < 100; ++i) {
    for (int round = 0; round < 4; round += 2) {
      dit8.setArgs(round, bufLong1, bufLongTmp);
      queue.run(dit8, GS, SIZE / 32);
      dit8.setArgs(round + 1, bufLongTmp, bufLong1);
      queue.run(dit8, GS, SIZE / 32);
    }
  }
  queue.finish();
  time("perf DIT8");
  

  /*
  sq4k.setArgs(buf1, buf2);
  for (int i = 0; i < 1000; ++i) {
    queue.run(sq4k, GS, words * GS / (64 * 64));
  }
  queue.finish();
  time("sq4k");
  */

  
  /*
  // Initial DIF round on zero-padded input.
  difIniZeropad.setArgs(bitsBuf, buf2);

  for (int i = 0; i < 100; ++i) {
  queue.run(difIniZeropad, GS, SIZE / 4);

  dif2.setArgs(10, buf2, buf1);
  queue.run(dif2, GS, SIZE / 2);
  
  for (int i = 0; i < 5; ++i) {
    dif2.setArgs(9 - i * 2, buf1, buf2);
    queue.run(dif2, GS, SIZE / 2);
    dif2.setArgs(8 - i * 2, buf2, buf1);
    queue.run(dif2, GS, SIZE / 2);
  }
  }
  queue.finish();
  time("dif1");
  */

  
  //difIniZeropadShifted.setArgs(bitsBuf, buf2);
  
  /*
  for (int i = 0; i < 5; ++i) {
    dit2.setArgs(i * 2, bigBuf, tmpBuf);
    queue.run(dit2, GS, SIZE / 2);
    dit2.setArgs(i * 2 + 1, tmpBuf, bigBuf);
    queue.run(dit2, GS, SIZE / 2);
  }
  queue.finish();
  time("dit2");
  
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
