#include <CL/cl.h>
#include <assert.h>
#include <stdio.h>
#include "time.h"

#define CHECK(err) { int e = err; if (e != CL_SUCCESS) { fprintf(stderr, "error %d\n", e); assert(false); }}

#define CHECK2(err, mes) { int e = err; if (e != CL_SUCCESS) { fprintf(stderr, "error %d (%s)\n", e, mes); assert(false); }}

class Context {
public:
  cl_device_id device;
  cl_context context;
  
  Context() {
    cl_platform_id platform;
    CHECK(clGetPlatformIDs(1, &platform, NULL));
    CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));
    int err;
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err); CHECK(err);
  }

  ~Context() {
    CHECK(clReleaseContext(context));
  }
};

class Program {
public:
  cl_program program;

  Program() : program(nullptr) { }

  Program(Context &c, const char *f) {
    compileCL2(c, f);
    time(f);
  }

  ~Program() {
    clReleaseProgram(program);
  }
  
  void compileCL1(Context &c, FILE *f) { init(c, f, false, false); }
  void compileCL2(Context &c, FILE *f) { init(c, f, false, true); }
  void compileCL1(Context &c, const char *f) { init(c, f, false, false); }
  void compileCL2(Context &c, const char *f) { init(c, f, false, true); }

  void loadCL1(Context &c, const char *f) { init(c, f, true, false); }
  void loadCL2(Context &c, const char *f) { init(c, f, true, true); }

  void dumpBin(FILE *fo) {
    size_t programSize;
    CHECK(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &programSize, NULL));
    char buf[programSize + 1];
    char *pbuf = buf;
    CHECK(clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(size_t), &pbuf, NULL));
    fwrite(buf, 1, programSize, fo);
    fflush(fo);
  }

 private:
  
  void init(Context &c, const char *fileName, bool isBinary, bool isCL2) {
    FILE *f = fopen(fileName, "r");
    assert(f);
    init(c, f, isBinary, isCL2);
    fclose(f);
  }
  
  void init(Context &c, FILE *f, bool isBinary, bool isCL2) {
    char buf[256 * 1024];
    size_t size = read(buf, sizeof(buf), f);
    char *pbuf = buf;
    int err;
    program = isBinary
      ? clCreateProgramWithBinary(c.context, 1, &(c.device), &size, (const unsigned char **) &pbuf, NULL, &err)
      : clCreateProgramWithSource(c.context, 1, (const char **)&pbuf, &size, &err);    
    CHECK(err);
    const char *opts = isCL2 ?
      "-O5 -Werror -cl-std=CL2.0 -cl-uniform-work-group-size -I. -fno-bin-llvmir -save-temps=tmp2/" :
      "-O2 -Werror -cl-fast-relaxed-math -I. -fno-bin-llvmir -fno-bin-source -fno-bin-amdil -save-temps=tmp1/";
    /*
    const char *opts = isCL2
      ? "-O2 -cl-std=CL2.0 -cl-uniform-work-group-size -fno-bin-source -fno-bin-llvmir -fno-bin-amdil -save-temps"
      : "-O2 -cl-fast-relaxed-math -fno-bin-llvmir -save-temps";
    */
    // 
    // -cl-std=CL2.0 -Werror 
    // -cl-uniform-work-group-size 
    // -fno-bin-source -fno-bin-llvmir -fno-bin-amdil"
    // -fno-bin-hsail #bad
    // -O0 -cl-opt-disable
    err = clBuildProgram(program, 1, &(c.device), opts, NULL, NULL);
    if (err != CL_SUCCESS) {
      size_t logSize;
      clGetProgramBuildInfo(program, c.device, CL_PROGRAM_BUILD_LOG, sizeof(buf), buf, &logSize);
      buf[logSize] = 0;
      fprintf(stderr, "log %s\n", buf);
    }
    CHECK(err);
  }
  
  size_t read(char *buf, int bufSize, FILE *f) {
    size_t size = fread(buf, 1, bufSize, f); assert(size);
    return size;
  }

  /*
  size_t read(char *buf, int bufSize, const char *fileName) {
    FILE *f = fopen(fileName, "r"); assert(f);
    size_t size = read(buf, bufSize, f);
    fclose(f);
    return size;
  }
  */
};

class Kernel {
public:
  cl_kernel k;
  
  Kernel(Program &p, const char *name) {
    int err;
    k = clCreateKernel(p.program, name, &err);
    CHECK2(err, name);
  }

  ~Kernel() {
    clReleaseKernel(k); k = 0;
  }

  template<class T> void setArg(int pos, const T &value) { CHECK(clSetKernelArg(k, pos, sizeof(value), &value)); }

  template<class A> void setArgs(const A &a) {
    setArg(0, a);
  }
  
  template<class A, class B> void setArgs(const A &a, const B &b) {
    setArgs(a);
    setArg(1, b);
  }
  
  template<class A, class B, class C> void setArgs(const A &a, const B &b, const C &c) {
    setArgs(a, b);
    setArg(2, c);
  }

  template<class A, class B, class C, class D> void setArgs(const A &a, const B &b, const C &c, const D &d) {
    setArgs(a, b, c);
    setArg(3, d);
  }
  
  template<class A, class B, class C, class D, class E> void setArgs(const A &a, const B &b, const C &c, const D &d, const E &e) {
    setArgs(a, b, c, d);
    setArg(4, e);
  }

  template<class A, class B, class C, class D, class E, class F> void setArgs(const A &a, const B &b, const C &c, const D &d, const E &e, const F &f) {
    setArgs(a, b, c, d, e);
    setArg(5, f);
  }
};

class Buf {
 public:
  cl_mem buf;

  Buf(Context &c, uint kind, size_t size, void *ptr) {
    int err;
    buf = clCreateBuffer(c.context, kind, size, ptr, &err);
    CHECK(err);
  }

  Buf(Context &c, uint kind, size_t size) :
  Buf(c, kind, size, NULL) { }

  ~Buf() { CHECK(clReleaseMemObject(buf)); }
};

class Queue {
public:
  cl_command_queue queue;

  Queue(Context &c) {
    int err;
    queue = clCreateCommandQueueWithProperties(c.context, c.device, 0, &err);
    // queue = clCreateCommandQueue(c.context, c.device, 0, &err);
    CHECK(err);
  }

  ~Queue() {
    flush();
    CHECK(clFinish(queue));
    CHECK(clReleaseCommandQueue(queue));
  }

  void time(const char *s) {
    finish();
    ::time(s);
  }
  
  void run(Kernel &k, size_t groupSize, size_t workSize) {
    // printf("size %lu %lu\n", groupSize, workSize);
    CHECK(clEnqueueNDRangeKernel(queue, k.k, 1, NULL, &workSize, &groupSize, 0, NULL, NULL));
  }

  void run2D(Kernel &k, size_t groupSize0, size_t groupSize1, size_t workSize0, size_t workSize1) {
    // printf("size %lu %lu\n", groupSize, workSize);
    size_t groupSizes[] = {groupSize0, groupSize1};
    size_t workSizes[] = {workSize0, workSize1};
    CHECK(clEnqueueNDRangeKernel(queue, k.k, 2, NULL, workSizes, groupSizes, 0, NULL, NULL));
  }

  /*
  template<class T> void read(Buf &buf, T &var) {
    CHECK(clEnqueueReadBuffer(queue, buf.buf, CL_BLOCKING, 0, sizeof(var), &var, 0, NULL, NULL));
  }
  */

  uint read(Buf &buf) {
    uint ret = 0;
    CHECK(clEnqueueReadBuffer(queue, buf.buf, CL_BLOCKING, 0, sizeof(uint), &ret, 0, NULL, NULL));
    return ret;
  }

  void readBlocking(Buf *buf, size_t start, size_t size, void *data) {
    CHECK(clEnqueueReadBuffer(queue, buf->buf, CL_BLOCKING, start, size, data, 0, NULL, NULL));
  }
  
  void write(Buf &buf, void *ptr, size_t size, bool blocking) {
    CHECK(clEnqueueWriteBuffer(queue, buf.buf, blocking ? CL_BLOCKING : CL_NON_BLOCKING,
                               0, size, ptr, 0, NULL, NULL));
  }
  
  void writeBlocking(Buf &buf, void *ptr, size_t size) { write(buf, ptr, size, true); }
  void writeAsync(Buf &buf, void *ptr, size_t size) { write(buf, ptr, size, false); }

  void write(Buf &buf, void *data, size_t size) {
    CHECK(clEnqueueWriteBuffer(queue, buf.buf, CL_NON_BLOCKING, 0, size, data, 0, NULL, NULL));
  }

  void zero(Buf &buf, size_t offset, size_t size) {
    unsigned zero = 0;
    CHECK(clEnqueueFillBuffer(queue, buf.buf, &zero, sizeof(zero), offset, size, 0, NULL, NULL));
  }

  /*
  template<class T> void write(Buf &buf, T &var, bool blocking) {
    CHECK(clEnqueueWriteBuffer(queue, buf.buf, blocking ? CL_BLOCKING : CL_NON_BLOCKING,
                               0, sizeof(var), &var, 0, NULL, NULL));
  }
  template<class T> void writeBlocking(Buf &buf, T &var) { write(buf, var, true); }
  template<class T> void writeAsync(Buf &buf, T &var) { write(buf, var, false); }
  */

  void flush() { CHECK(clFlush(queue)); }

  void finish() { CHECK(clFinish(queue)); }

  void barrier() {
    CHECK(clEnqueueBarrierWithWaitList(queue, 0, NULL, NULL));
  }
};
