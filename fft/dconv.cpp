#include "clutil.h"
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <memory>

#define SIZE (8 * 1024 * 1024)
#define GS 256

#define K(program, name) Kernel name(program, #name);

Context c;
Queue queue(c);
Program program(c, "dconv.cl");

K(program, dif_0);
K(program, dif_3);
K(program, dif_6);
K(program, dif_9);

K(program, dit_0);
K(program, dit_3);
K(program, dit_6);
K(program, dit_9);

K(program, round0);  

void dif8(Queue &queue, Buf &buf, Buf &tmp) {
  dif_9.setArgs(buf, tmp);
  queue.run(dif_9, GS, SIZE / 32);
  dif_6.setArgs(tmp, buf);
  queue.run(dif_6, GS, SIZE / 32);
  dif_3.setArgs(buf, tmp);
  queue.run(dif_3, GS, SIZE / 32);
  dif_0.setArgs(tmp, buf);
  queue.run(dif_0, GS, SIZE / 32);
}

void dit8(Queue &queue, Buf &buf, Buf &tmp) {
  dit_0.setArgs(buf, tmp);
  queue.run(dit_0, GS, SIZE / 32);
  dit_3.setArgs(tmp, buf);
  queue.run(dit_3, GS, SIZE / 32);
  dit_6.setArgs(buf, tmp);
  queue.run(dit_6, GS, SIZE / 32);
  dit_9.setArgs(tmp, buf);
  queue.run(dit_9, GS, SIZE / 32);
}

int main(int argc, char **argv) {
  time("main entry");
  
  double *data = new double[SIZE];
  
  srandom(0);
  for (int i = 0; i < SIZE; ++i) { data[i] = (random() & 0xffffff) - (1 << 23); }
  time("random");
  
  Buf buf1(c, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * SIZE, data);
  Buf buf2(c, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * SIZE, data);
  Buf bufTmp(c, CL_MEM_READ_WRITE, sizeof(double) * SIZE, 0);
  time("alloc gpu buffers");

  dif8(queue, buf1, bufTmp);
  queue.time("dif8");
  dit8(queue, buf1, bufTmp);
  queue.time("dit8");

  /*
  dif_3.setArgs(buf1, bufTmp);
  queue.run(dif_3, GS, SIZE / 32);
  dit_3.setArgs(bufTmp, buf1);
  queue.run(dit_3, GS, SIZE / 32);
  */
  double *data2 = new double[SIZE];
  queue.readBlocking(&buf1, 0, sizeof(double) * SIZE, data2);
  time("read");
  int nerr = 0;
  for (int i = 0; i < SIZE; ++i) {
    double b = data2[i] / (4 * 1024);
    if (data[i] != b) {
      printf("%d %f %f\n", nerr, data[i], b);
      ++nerr;
      if (nerr >= 10) {
        break;
      }
    }
  }
  exit(0);

  
  round0.setArgs(buf1, bufTmp);
  queue.run(round0, GS, SIZE / 2);
  queue.time("warm-up");

  for (int i = 0; i < 500; ++i) {
    round0.setArgs(buf1, bufTmp);
    queue.run(round0, GS, SIZE / 2);
    round0.setArgs(bufTmp, buf1);
    queue.run(round0, GS, SIZE / 2);
  }
  queue.time("round0");

  dif_3.setArgs(buf1, bufTmp);
  queue.run(dif_3, GS, SIZE / 32);
  queue.time("warm-up");

  for (int i = 0; i < 500; ++i) {
    dif_3.setArgs(buf1, bufTmp);
    queue.run(dif_3, GS, SIZE / 32);
    dif_3.setArgs(bufTmp, buf1);
    queue.run(dif_3, GS, SIZE / 32);
  }
  queue.time("dif_3");

  dit_3.setArgs(buf1, bufTmp);
  queue.run(dit_3, GS, SIZE / 32);
  queue.time("warm-up");

  for (int i = 0; i < 500; ++i) {
    dit_3.setArgs(buf1, bufTmp);
    queue.run(dit_3, GS, SIZE / 32);
    dit_3.setArgs(bufTmp, buf1);
    queue.run(dit_3, GS, SIZE / 32);
  }
  queue.time("dit_3");


  
  
#if 0
  
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
      // printf("%d %d %ld\n", i, data[i], tmpLong1[i]);
      if (++err >= 10) { exit(1); }
    }
  }  
  time("OK FFT radix8 round-trip");

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
    for (int round = 0; round < 4; round += 2) {
      dit8.setArgs(round, bufLong1, bufLongTmp);
      queue.run(dit8, GS, SIZE / 32);
      dit8.setArgs(round + 1, bufLongTmp, bufLong1);
      queue.run(dit8, GS, SIZE / 32);
    }
  }
  queue.finish();
  time("perf DIT8");

#endif
}
