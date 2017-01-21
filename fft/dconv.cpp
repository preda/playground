#include "clutil.h"
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <memory>

#define SIZE (4 * 1024 * 1024)
#define GS 256

#define K(program, name) Kernel name(program, #name);

Context c;
Queue queue(c);
Program program(c, "dconv.cl");

K(program, dif_0);

K(program, round0);
K(program, copy);
K(program, dif);
K(program, dit);
K(program, conv4k);
K(program, convfft);
K(program, dif4k);

void dif8a(Queue &queue, Buf &buf1, Buf &buf2, unsigned size) {
  dif.setArgs(6, buf1, buf2);
  queue.run(dif, GS, size / 32);
  dif.setArgs(3, buf2, buf1);
  queue.run(dif, GS, size / 32);
  dif_0.setArgs(buf1, buf2);
  queue.run(dif_0, GS, size / 32);
  // dif.setArgs(0, tmp, buf);
  // queue.run(dif, GS, size / 32);  
}

void dit8a(Queue &queue, Buf &buf1, Buf &buf2, unsigned size) {
  dit.setArgs(0, buf1, buf2);
  queue.run(dit, GS, size / 32);
  dit.setArgs(3, buf2, buf1);
  queue.run(dit, GS, size / 32);
  dit.setArgs(6, buf1, buf2);
  queue.run(dit, GS, size / 32);
}


int main(int argc, char **argv) {
  time("main entry");
  
  double *data = new double[SIZE * 2];
  
  srandom(0);
  for (int i = 0; i < SIZE; ++i) { data[i] = (random() & 0xffffff) - (1 << 23); }
  time("random");
  
  Buf buf1(c, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * SIZE * 2, data);
  Buf buf2(c, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * SIZE * 2, data);
  Buf bufTmp(c, CL_MEM_READ_WRITE, sizeof(double) * SIZE * 2, 0);
  time("alloc gpu buffers");
  
  /*
  dif8a(queue, buf1, bufTmp, SIZE);
  queue.time("dif8");
  dit8a(queue, bufTmp, buf1, SIZE);
  queue.time("dit8");
  */

  /*
  double *data2 = new double[SIZE];
  queue.readBlocking(&bufTmp, 0, sizeof(double) * SIZE, data2);
  time("read");
  int nerr = 0;
  for (int i = 0; i < SIZE; ++i) {
    double b = data2[i] / 64;
    if (data[i] != b) {
      printf("%d %f %f\n", i, data[i], b);
      ++nerr;
      if (nerr >= 10) {
        break;
      }
    }
  }
  */

  conv4k.setArgs(buf1, bufTmp);
  queue.run(conv4k, GS, SIZE / 16);
  queue.time("");

  for (int i = 0; i < 1000; ++i) {
    queue.run(conv4k, GS, SIZE / 16);
  }
  queue.time("conv4k");


  dif4k.setArgs(buf1);
  queue.run(dif4k, GS, SIZE / 16);
  queue.time("");
  
  for (int i = 0; i < 1000; ++i) {
    queue.run(dif4k, GS, SIZE / 16);
  }
  queue.time("dif4k");
  
  convfft.setArgs(buf1);
  queue.run(convfft, GS, SIZE / 16);
  queue.time("");
  
  for (int i = 0; i < 1000; ++i) {
    queue.run(convfft, GS, SIZE / 16);
  }
  queue.time("convfft");
  
  exit(0);

  
  
  for (int i = 0; i < 1000; ++i) { dif8a(queue, buf1, bufTmp, SIZE); }
  queue.time("dif");

  for (int i = 0; i < 1000; ++i) { dit8a(queue, buf1, bufTmp, SIZE); }
  queue.time("dit");

  
  for (int i = 0; i < 2000; ++i) {
    round0.setArgs(buf1, bufTmp);
    queue.run(round0, GS, SIZE / 2);
    round0.setArgs(bufTmp, buf1);
    queue.run(round0, GS, SIZE / 2);
  }
  queue.time("round0");

  for (int i = 0; i < 2000; ++i) {
    copy.setArgs(buf1, bufTmp);
    queue.run(copy, GS, SIZE);
    copy.setArgs(bufTmp, buf1);
    queue.run(copy, GS, SIZE);
  }
  queue.time("copy");

}
