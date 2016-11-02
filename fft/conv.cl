#include "template.h"
FUNCS(int)
FUNCS(long)

#define GS 256
#define KERNEL(groupSize) kernel __attribute__((reqd_work_group_size(groupSize, 1, 1)))

void bar() { barrier(CLK_LOCAL_MEM_FENCE); }

unsigned cut(unsigned x) { return x & 0x3fffffff; }

int read(global int *in, int width, int line, int p) {
  return in[cut(line * width + p)];
}

int readZeropad(global int *in, int width, int line, int p) {
  return (p < width/2) ? in[cut(line * (width/2) + p)] : 0;
}

void write(int u, global int *out, int width, int line, int p) {
  out[cut(line * width + p)] = u;
}

int readShifted(global int *in, int width, int line, int p) {
  int u = read(in, width, line, p & (width - 1));
  return (p < width) ? u : -u;
}

int readZeropadShifted(global int *in, int width, int line, int p) {
  int u = readZeropad(in, width, line, p & (width - 1));
  return (p < width) ? u : -u;
}

void writeShifted(int u, global int *out, int width, int line, int p) {
  write((p < width) ? u : -u, out, width, line, p & (width - 1));
}


// DIF step for a 2**12 FFT.
// round goes down from N-1 to 0.
KERNEL(GS) void difStep(int round, global int *in, global int *out) {
  const int N = 12;
  int width = 1 << N;
  uint groupsPerLine = width / GS;
  uint g = get_group_id(0) / groupsPerLine;
  uint k = get_group_id(0) & (groupsPerLine - 1);

  uint mr = 1 << round;
  uint j = g & (mr - 1);
  uint r = (g & ~(mr - 1)) << 1;
  uint e = j << (N - 1 - round);
  uint p = get_local_id(0) + k * GS;
  
  int u0 = read(in, width, j + r,      p);
  int u1 = read(in, width, j + r + mr, p);
  write(       u0 + u1, out, width, j + r,      p);
  writeShifted(u0 - u1, out, width, j + r + mr, p + e);
}

KERNEL(GS) void difIniZeropad(global int *in, global int *out) {
  const int N = 12;
  int width = 1 << N;
  uint groupsPerLine = width / GS;
  uint g = get_group_id(0) / groupsPerLine;
  uint k = get_group_id(0) & (groupsPerLine - 1);

  uint mr = 1 << (N - 1);
  uint j = g;
  uint e = j;
  uint p = get_local_id(0) + k * GS;
  
  int u0 = readZeropad(in, width, j,      p);
  int u1 = readZeropad(in, width, j + mr, p);
  write(       u0 + u1, out, width, j,      p);
  writeShifted(u0 - u1, out, width, j + mr, p + e);
}

KERNEL(GS) void difIniZeropadShifted(global int *in, global int *out) {
  const int N = 12;
  int width = 1 << N;
  uint groupsPerLine = width / GS;
  uint g = get_group_id(0) / groupsPerLine;
  uint k = get_group_id(0) & (groupsPerLine - 1);

  uint mr = 1 << (N - 1);
  uint j = g;
  uint e = j;
  uint p = get_local_id(0) + k * GS;
  
  int u0 = readZeropadShifted(in, width, j,      p + j);
  int u1 = readZeropadShifted(in, width, j + mr, p + j + mr);
  write(       u0 + u1, out, width, j,      p);
  writeShifted(u0 - u1, out, width, j + mr, p + e);
}

// DIT step for a 2**12 FFT
// round goes up from 0 to N-1.
KERNEL(GS) void ditStep(int round, global int *in, global int *out) {
  const int N = 12;
  int width = 1 << N;
  uint groupsPerLine = width / GS;
  uint g = get_group_id(0) / groupsPerLine;
  uint k = get_group_id(0) & (groupsPerLine - 1);

  uint mr = 1 << round;
  uint j = g & (mr - 1);
  uint r = (g & ~(mr - 1)) << 1;
  uint e = j << (N - 1 - round);
  uint p = get_local_id(0) + k * GS;

  int u0 = read(       in, width, j + r,      p);
  int u1 = readShifted(in, width, j + r + mr, p + e);
  write(halfAdd(u0, u1),  out, width, j + r,      p);
  write(halfAdd(u0, -u1), out, width, j + r + mr, p);
}

KERNEL(GS) void ditFinalShifted(global int *in, global int *out) {
  const int N = 12;
  int width = 1 << N;
  uint groupsPerLine = width / GS;
  uint g = get_group_id(0) / groupsPerLine;
  uint k = get_group_id(0) & (groupsPerLine - 1);

  uint mr = 1 << (N - 1);
  uint j = g;
  uint e = j;
  uint p = get_local_id(0) + k * GS;

  int u0 = read(       in, width, j,      p);
  int u1 = readShifted(in, width, j + mr, p + e);
  writeShifted(halfAdd(u0, u1),  out, width, j,      p + j);
  writeShifted(halfAdd(u0, -u1), out, width, j + mr, p + mr);
}

KERNEL(GS) void sq4k(global int *in, global int *out) {
  // Each group handles one contiguous line of 4k ints, stored in LDS.
  local int lds[64 * 64];
  
  // First, load 4k ints from global into LDS, transposed. Warp to avoid bank conflicts.
  for (int i = 0; i < 16; ++i) {
    uint line = get_local_id(0) & 63;
    lds[line * 64 + ((i*4 + (get_local_id(0) >> 6) + line) & 63)] = in[get_group_id(0) * 64 * 64 + get_local_id(0) + i * 64 * 4];
  }

  //De-warp.
  bar();
  for (int i = 0; i < 16; ++i) {
    uint line = i * 4 + (get_local_id(0) >> 6);
    lds[i * 64 * 4 + get_local_id(0)] = lds[line * 64 + ((line + get_local_id(0)) & 63)];
  }

  bar();
  for (int i = 0; i < 16; ++i) {
    out[get_group_id(0) * 64 * 64 + i * 64 * 4 + get_local_id(0)] = lds[i * 64 * 4 + get_local_id(0)];
  }
}


int2 sumdiff(int x, int y) { return (int2) (x + y, x - y); }

// 1 int x int -> long multiply
long mul(int x, int y) { return x * (long) y; }

// Below, "sq" means "negacyclic auto convolution" (which is somewhat related to squaring).
// 2 muls
long2 sq2(int2 v) { return (long2) (mul(v.x + v.y, v.x - v.y), mul(v.x, v.y) * 2); }

// 6 muls (3 x sq2())
long4 sq4(int4 v) {
  long2 x2 = sq2(v.xz);
  long2 y2 = sq2(v.yw);
  return ((long4) (x2 + shift(y2, 1), x2 + y2 - sq2(v.x - v.y))).xzyw;
}

// 18 muls (3 x sq4())
long8 sq8(int8 v) {
  int4 x = v.s0246;
  int4 y = v.s1357;
  long4 x2 = sq4(x);
  long4 y2 = sq4(y);
  return ((long8) (x2 + shift(y2, 1), x2 + y2 - sq4(x - y))).s04152637;
}

/*
kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void negaconv4k(global int *in, global long *out) {
  local long lds[2 * 8 * 128];
  local int *ids = (local int *)lds;
  uint line = get_group_id(0);
  uint STRIDE = 129;
  for (int i = 0; i < 8; ++i) {
    ids[(get_local_id(0) & 7) * STRIDE + (get_local_id(0) >> 3) + i * 16] = in[line * 1024 + i * 128 + get_local_id(0)];
  }
  for (int i = 0; i < 8; ++i) {
    int x = ids[i * STRIDE + get_local_id(0)];
    uint p = get_local_id(0) + i * 16;
    ids[(i + 8) * STRIDE + (p & 127)] = (p < 128) ? x : -x;
  }
  // DIF 3 rounds. See difStep().
  for (int round = 2; round >= 0; --round) {
    uint mr = 1 << round;
    for (uint i = 0; i < 8; ++i) {
      uint j = i & (mr - 1);
      uint r = (i & ~(mr - 1)) << 1;
      uint e = (j << (4 - 1 - round)) * 16;
      uint p0 = (j + r) * STRIDE + get_local_id(0);
      uint p1 = get_local_id(0) + e;
      
      barrier(CLK_LOCAL_MEM_FENCE);
      int x = ids[p0];
      int y = ids[p0 + mr * STRIDE];
      
      barrier(CLK_LOCAL_MEM_FENCE);
      ids[p0] = x + y;
      y = x - y;
      ids[(j + r + mr) * STRIDE + (p1 & 127)] = (p1 < 128) ? y : -y;
    }
  }

  uint line = get_local_id(0) >> 3;
  uint k = get_local_id(0) & 7;
  
  for (uint i = 0; i < 16; ++i) {
    
  }
}
*/

// 8 x sq8(), 148 muls
kernel __attribute__((reqd_work_group_size(64, 1, 1))) void negaconv64(global int *in, global long *out) {  
  local int lds[16 * 8 * 2];

  uint group = get_group_id(0);
  for (int rep = 0; rep < 8; ++rep) {
    int v = in[group * 64 * 8 + get_local_id(0)];
    lds[rep * 128 + get_local_id(0)] = v;
    uint col = get_local_id(0) & 7;
    uint p = get_local_id(0) >> 3;
    lds[rep * 128 + 64 + (get_local_id(0) & 7) + ((p + col) & 7) * 8] = (p + col < 8) ? v : -v;
  }
  for (int round = 2; round >= 0; --round) {
    
  }
  
  
  /*
  int v = in[get_global_id(0)];
  uint line = (get_local_id(0) & 7);
  uint p = get_local_id(0) >> 3;
  lds[line * 8 + p] = x;
  lds[64 + line * 8 + (p + line) & 7] = (p + line < 8) ? v : -v;
  for (int round = 2; round >= 0; --round) {
    uint g = get_local_id(0) >> 3;
    uint k = get_local_id(0) & 7;
    uint mr = 1 << round;
    uint j = g & (mr - 1);
    uint r = (g & ~(mr - 1)) << 1;
    uint e = j << (3 - round);
    uint p0 = (j + r) * 8 + k;
    int u0 = lds[p0];
    int u1 = lds[p0 + mr * 8];
    lds[p0] = u0 + u1;
    u1 = u0 - u1;
    uint p1 = k + e;
    lds[(j + r + mr) * 8 + (p1 & 7)] = (p1 < 8) ? u1 : -u1;
  }
  */
  
}




void set(int4 *outa, int4 *outb, int4 a, int4 b) {
  *outa = a;
  *outb = b;
}

long4 halfAdd4(long4 a, long4 b) {
  return (long4) (halfAdd(a.x, b.x), halfAdd(a.y, b.y), halfAdd(a.z, b.z), halfAdd(a.w, b.w));
}

void halfAddSub(long4 *a, long4 *b) {
  long4 t = *b;
  *b = halfAdd4(*a, -t);
  *a = halfAdd4(*a, t);
}
