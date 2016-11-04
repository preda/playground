#include "template.h"

unsigned cut(unsigned x) { return x & 0x3fffffff; }

FUNCS(int)
FUNCS(long)

#define GS 256
#define KERNEL(groupSize) kernel __attribute__((reqd_work_group_size(groupSize, 1, 1)))

void bar() { barrier(CLK_LOCAL_MEM_FENCE); }

int readZeropad(global int *in, int width, int line, int p) {
  return (p < width/2) ? in[cut(line * (width/2) + p)] : 0;
}

int readShifted(global int *in, int width, int line, int p) {
  int u = read(in, width, line, p & (width - 1));
  return (p < width) ? u : -u;
}

int readZeropadShifted(global int *in, int width, int line, int p) {
  int u = readZeropad(in, width, line, p & (width - 1));
  return (p < width) ? u : -u;
}

void writeShifted(int u, global int *out, int lw, int line, int p) {
  write((p & (1 << lw)) ? -u : u, out, 1 << lw, line, p & ((1 << lw) - 1));
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
  uint e = j << (N + 1 - (round + 1));
  uint line = j + r;
  uint p = get_local_id(0) + k * GS;
  
  int u0 = read(in, width, line,      p);
  int u1 = read(in, width, line + mr, p);
  write(       u0 + u1, out, width, line,      p);
  writeShifted(u0 - u1, out, N, line + mr, p + e);
}

int2 read2(global int *in, int width, int line, int p) {
  return (int2) (read(in, width, line, p), readShifted(in, width, line, p + width/2));
}

void write2(int2 u, global int *out, int lw, int line, int p) {
  writeShifted(u.x, out, lw, line, p);
  writeShifted(u.y, out, lw, line, p + (1 << (lw - 1)));
}

#define ADDSUB(a, b) { int2 tmp = a; a = tmp + b; b = tmp - b; }

// Radix-4 DIF step for a 2**12 FFT.
// round goes down by 1 from 5 to 0.
KERNEL(GS) void dif4Step(int round, global int *in, global int *out) {
  const int N = 12;
  int width = 1 << N;
  uint groupsPerLine = width / (GS * 2);
  uint k = get_group_id(0) & (groupsPerLine - 1);
  uint g = get_group_id(0) / groupsPerLine;

  uint mr = 1 << (round * 2);
  uint j = g & (mr - 1);
  uint r = (g & ~(mr - 1)) << 2;
  uint e = j << (N + 1 - 2 * (round + 1));
  uint line = j + r;
  uint p = get_local_id(0) + k * GS;

  int2 u0 = read2(in, width, line, p);
  int2 u2 = read2(in, width, line + mr * 2, p);
  ADDSUB(u0, u2);
  
  int2 u1 = read2(in, width, line + mr, p);
  int2 u3 = read2(in, width, line + mr * 3, p);
  ADDSUB(u1, u3);
  
  write2(u0 + u1, out, N, line, p);
  write2(u0 - u1, out, N, line + mr, p + e * 2);
  
  u3 = shift(u3, 1);
  write2(u2 + u3, out, N, line + mr * 2, p + e);
  write2(u2 - u3, out, N, line + mr * 3, p + e * 3);
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
  writeShifted(u0 - u1, out, N, j + mr, p + e);
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
  writeShifted(u0 - u1, out, N, j + mr, p + e);
}

// DIT step for a 2**12 FFT
// round goes up from 0 to N-1.
KERNEL(GS) void ditStep(int round, global long *in, global long *out) {
  const int N = 12;
  int width = 1 << N;
  uint groupsPerLine = width / GS;
  uint g = get_group_id(0) / groupsPerLine;
  uint k = get_group_id(0) & (groupsPerLine - 1);

  uint mr = 1 << round;
  uint j = g & (mr - 1);
  uint r = (g & ~(mr - 1)) << 1;
  uint e = j << (N + 1 - (round + 1));
  uint line = j + r;
  uint p = get_local_id(0) + k * GS;

  long u0 = read(       in, width, line,      p);
  long u1 = readShifted(in, width, line + mr, p + e);
  write(halfAdd(u0, u1),  out, width, line,      p);
  write(halfAdd(u0, -u1), out, width, line + mr, p);
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
  writeShifted(halfAdd(u0, u1),  out, N, j,      p + j);
  writeShifted(halfAdd(u0, -u1), out, N, j + mr, p + mr);
}

void ldsWriteShifted(int u, local int *lds, int width, uint line, uint p) {
  lds[line * width + (p & (width - 1))] = (p < width) ? u : -u;
}

void ldsWriteWarped(int u, local int *lds, int width, uint line, uint p) {
  lds[line * width + ((p + line) & (width - 1))] = u;
}

KERNEL(GS) void sq4k(global int *in, global int *out) {
  // Each group handles one contiguous line of 4k ints, stored in LDS.
  local int lds[64 * 64];
  
  // First, load 4k ints from global into LDS, transposed. Warp to avoid bank conflicts.
  for (int i = 0; i < 16; ++i) {
    uint line = get_local_id(0) & 63;
    lds[line * 64 + ((i*4 + (get_local_id(0) >> 6) + line) & 63)] =
      in[get_group_id(0) * 64 * 64 + i * 64 * 4 + get_local_id(0)];
  }

  //De-warp.
  bar();
  for (int i = 0; i < 16; ++i) {
    uint line = i * 4 + (get_local_id(0) >> 6);
    lds[i * 64 * 4 + get_local_id(0)] = lds[line * 64 + ((line + get_local_id(0)) & 63)];
  }

  // 6 rounds of DIF FFT. The last round applies warp.
  uint p = get_local_id(0) & 63;
  for (int round = 5; round >= 0; --round) {
    uint mr = 1 << round;
    bar();
    for (int i = 0; i < 8; ++i) {
      uint g = (get_local_id(0) >> 6) + i * 4;
      uint j = g & (mr - 1);
      uint line = j + ((g & ~(mr - 1)) << 1);
      uint e = j << (5 - round);
      int u0 = lds[line * 64 + p];
      int u1 = lds[(line + mr) * 64 + p];
      if (round == 0) { // last round, apply warp. No shift needed because e==0.
        ldsWriteWarped(u0 + u1, lds, 64, line, p);
        ldsWriteWarped(u0 - u1, lds, 64, line + mr, p);
      } else {
        lds[line * 64 + p] = u0 + u1;
        ldsWriteShifted(u0 - u1, lds, 64, line + mr, p + e);
      }
    }
  }

  // transpose LDS, write to global. Warp avoids bank conflicts.
  bar();
  for (int i = 0; i < 16; ++i) {
    out[get_group_id(0) * 64 * 64 + i * 64 * 4 + get_local_id(0)] =
      lds[(get_local_id(0) & 63) * 64 + ((get_local_id(0) & 63) + (get_local_id(0) >> 6) + i * 4) & 63];
  }

  /*
  bar();
  for (int i = 0; i < 16; ++i) {
    out[get_group_id(0) * 64 * 64 + i * 64 * 4 + get_local_id(0)] = lds[i * 64 * 4 + get_local_id(0)];
  }
  */
}

/*
KERNEL(64) void sq64(global int *in, global long *out) {
  local long ldsl[128];
  local int *lds = (local int *) ldsl;
  int u = in[get_global_id(0) * 64 + get_local_id(0)];
  uint line = get_local_id(0) & 7;
  ldsWriteWarped(u, lds, 8, line, get_local_id(0) >> 3);
  ldsWriteShifted(u, lds + 64, 8, line, (get_local_id(0) >> 3) + line);

  bar();
  line = get_local_id(0) >> 3;
  uint p = get_local_id(0) & 7;
  lds[line * 8 + p] = lds[line * 8 + ((p + line) & 7)];
  ldsWriteShifted(
}
*/

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