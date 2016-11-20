#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#endif

#define _OVL __attribute__((overloadable))
#define KERNEL(groupSize) kernel __attribute__((reqd_work_group_size(groupSize, 1, 1)))

#include "template.h"

#define ADDSUBI(a, b) {  int2 tmp = a; a = tmp + b; b = tmp - b; }
#define ADDSUBH(a, b)  { long2 tmp = a; a = (tmp + b) >> 1; b = (tmp - b) >> 1; }
#define ADDSUB4H(a, b) { long4 tmp = a; a = (tmp + b) >> 1; b = (tmp - b) >> 1; }
#define ADDSUB4D(a, b) { double4 tmp = a; a = tmp + b; b = tmp - b; }
#define ADDSUB4(a, b) { int4 tmp = a; a = tmp + b; b = tmp - b; }
#define SHIFT(u, e) u = shift(u, e);

#define GS 256

#define FFT_SETUP_CORE(N, round, radixExp, iniG) \
uint g = iniG; \
uint mr = 1 << (round * radixExp);\
uint j = g & (mr - 1);\
uint r = (g & ~(mr - 1)) << radixExp;\
uint e = j << (N + 1 - (round + 1) * radixExp);\
uint line = j + r;

#define FFT_SETUP(iniN, radixExp) \
uint N = iniN; \
uint groupsPerLine = (1 << (N - (radixExp - 1))) / GS;  \
FFT_SETUP_CORE(N, round, radixExp, get_group_id(0) / groupsPerLine);    \
uint k = get_group_id(0) & (groupsPerLine - 1);\
uint p = get_local_id(0) + k * GS;

void bar() { barrier(CLK_LOCAL_MEM_FENCE); }

unsigned cut4(unsigned x) { return x & 0x3fffffff; }
unsigned cut8(unsigned x) { return x & 0x1fffffff; }

int    _OVL read(global int    *in, uint N, uint line, uint p) { return in[cut4((line << N) + p)]; }
long   _OVL read(global long   *in, uint N, uint line, uint p) { return in[cut8((line << N) + p)]; }
double _OVL read(global double *in, uint N, uint line, uint p) { return in[cut8((line << N) + p)]; }
double _OVL read(local  double *in, uint N, uint line, uint p) { return in[(line << N) + p]; }

void _OVL write(int    u, global int    *out, uint N, uint line, uint p) { out[cut4((line << N) + p)] = u; }
void _OVL write(long   u, global long   *out, uint N, uint line, uint p) { out[cut8((line << N) + p)] = u; }
void _OVL write(double u, global double *out, uint N, uint line, uint p) { out[cut8((line << N) + p)] = u; }
void _OVL write(double u, local  double *out, uint N, uint line, uint p) { out[(line << N) + p] = u; }

FUNCS_SHIFT(int)
FUNCS_SHIFT(long)
FUNCS_SHIFT(double)

FUNCS_RW(int,    global)
FUNCS_RW(long,   global)
FUNCS_RW(double, global)
FUNCS_RW(double, local)

#define QW (1 << (N - 2))

int4 _OVL read4(global int *in, uint N, uint line, uint p) {
  int u[4];
  for (int i = 0; i < 4; ++i) { u[i] = readC(in, N, line, p + QW * i); }
  return (int4)(u[0], u[1], u[2], u[3]);
}

void _OVL write4(int4 u, global int *out, uint N, uint line, uint p) {
  for (int i = 0; i < 4; ++i) { writeC((int[4]){u.x, u.y, u.z, u.w}[i], out, N, line, p + QW * i); }
}

long4 _OVL read4(global long *in, uint N, uint line, uint p) {
  long u[4];
  for (int i = 0; i < 4; ++i) { u[i] = readC(in, N, line, p + QW * i); }
  return (long4)(u[0], u[1], u[2], u[3]);
}

void _OVL write4(long4 u, global long *out, uint N, uint line, uint p) {
  for (int i = 0; i < 4; ++i) { writeC((long[4]){u.x, u.y, u.z, u.w}[i], out, N, line, p + QW * i); }
}

void _OVL write4NC(long4 u, global long *out, uint N, uint line, uint p) {
  for (int i = 0; i < 4; ++i) { write((long[4]){u.x, u.y, u.z, u.w}[i], out, N, line, p + QW * i); }
}

double4 _OVL read4(global double *in, uint N, uint line, uint p) {
  double u[4];
  for (int i = 0; i < 4; ++i) { u[i] = readC(in, N, line, p + QW * i); }
  return (double4)(u[0], u[1], u[2], u[3]);
}

long4 _OVL read4NC(global long *in, uint N, uint line, uint p) {
  long u[4];
  for (int i = 0; i < 4; ++i) { u[i] = read(in, N, line, p + QW * i); }
  return (long4)(u[0], u[1], u[2], u[3]);
}

double4 _OVL read4NC(global double *in, uint N, uint line, uint p) {
  double u[4];
  for (int i = 0; i < 4; ++i) { u[i] = read(in, N, line, p + QW * i); }
  return (double4)(u[0], u[1], u[2], u[3]);
}

void _OVL write4(double4 u, global double *out, uint N, uint line, uint p) {
  for (int i = 0; i < 4; ++i) { writeC((double[4]){u.x, u.y, u.z, u.w}[i], out, N, line, p + QW * i); }
}

void _OVL write4NC(double4 u, global double *out, uint N, uint line, uint p) {
  for (int i = 0; i < 4; ++i) { write((double[4]){u.x, u.y, u.z, u.w}[i], out, N, line, p + QW * i); }
}

#undef QW

int readZeropad(global int *in, int N, int line, int p) { return (p & (1 << (N - 1))) ? 0 : readC(in, N - 1, line, p); }

// Radix-2 DIF step for a 2**12 FFT. round is 11 to 0.
KERNEL(GS) void dif2(int round, global int *in, global int *out) {
  FFT_SETUP(12, 1);
  
  int u0 =   read(in, N, line,      p);
  int u1 =   read(in, N, line + mr, p);
  write( u0 + u1, out, N, line,      p);
  writeC(u0 - u1, out, N, line + mr, p + e);
}

// Radix-2 DIT step for a 2**12 FFT. Round is 0 to 11.
KERNEL(GS) void dit2(int round, global long *in, global long *out) {
  FFT_SETUP(12, 1);

  long u0 = read( in, N, line,      p);
  long u1 = readC(in, N, line + mr, p + e);
  write((u0 + u1) >> 1, out, N, line,      p);
  write((u0 - u1) >> 1, out, N, line + mr, p);
}

// Radix-4 DIF step for a 2**12 FFT. Round is 5 to 0.
KERNEL(GS) void dif4(int round, global int *in, global int *out) {
  FFT_SETUP(12, 2);
  
  int2 u0 = read2(in, N, line,          p);
  int2 u1 = read2(in, N, line + mr,     p);
  int2 u2 = read2(in, N, line + mr * 2, p);
  int2 u3 = read2(in, N, line + mr * 3, p);
  
  ADDSUBI(u0, u2);
  ADDSUBI(u1, u3);
  
  write2( u0 + u1, out, N, line,      p);
  write2C(u0 - u1, out, N, line + mr, p + e * 2);
  
  u3 = shift(u3, 1);
  write2C(u2 + u3, out, N, line + mr * 2, p + e);
  write2C(u2 - u3, out, N, line + mr * 3, p + e * 3);
}

// Radix-4 DIT step for a 2**12 FFT. Round is 0 to 5.
KERNEL(GS) void dit4(int round, global long *in, global long *out) {
  FFT_SETUP(12, 2);

  long2 u0 = read2(in,  N, line,      p);
  long2 u2 = read2C(in, N, line + mr, p + e * 2);  
  long2 u1 = read2C(in, N, line + mr * 2, p + e);
  long2 u3 = read2C(in, N, line + mr * 3, p + e * 3);
  ADDSUBH(u0, u2);
  ADDSUBH(u1, u3);

  u3 = shift(u3, -1);
  write2((u0 + u1) >> 1, out, N, line,          p);
  write2((u0 - u1) >> 1, out, N, line + mr * 2, p);
  write2((u2 + u3) >> 1, out, N, line + mr,     p);
  write2((u2 - u3) >> 1, out, N, line + mr * 3, p);
}

// Radix-8 DIF step. round is 3 to 0.
KERNEL(GS) void dif8(int round, global int *in, global int *out) {
  FFT_SETUP(12, 3);
  int4 u[8];
  uint revbin[8] = {0, 4, 2, 6, 1, 5, 3, 7};

  for (int i = 0; i < 8; ++i) { u[i] = read4(in, N, line + mr * i, p); }
  for (int i = 0; i < 4; ++i) { ADDSUB4(u[i], u[i + 4]); }
  for (int i = 1; i < 4; ++i) { SHIFT(u[i + 4], i); }

  for (int i = 0; i < 8; i += 4) {
    ADDSUB4(u[0 + i], u[2 + i]);
    ADDSUB4(u[1 + i], u[3 + i]);
    SHIFT(u[3 + i], 2);
  }

  for (int i = 0; i < 8; i += 2) {
    write4(u[i] + u[i + 1], out, N, line + mr * i,       p + e * revbin[i]);
    write4(u[i] - u[i + 1], out, N, line + mr * (i + 1), p + e * revbin[i + 1]);
  }
}

// Radix-8 DIT step. round is 0 to 3.
KERNEL(GS) void dit8(int round, global long *in, global long *out) {
  FFT_SETUP(12, 3);
  long4 u[8];
  uint revbin[8] = {0, 4, 2, 6, 1, 5, 3, 7};

  u[0] = read4NC(in, N, line, p);
  for (int i = 1; i < 8; ++i) { u[i] = read4(in, N, line + mr * revbin[i], p + e * i); }
  for (int i = 0; i < 4; ++i) { ADDSUB4H(u[i], u[i + 4]); }
  for (int i = 1; i < 4; ++i) { SHIFT(u[i + 4], -i); }

  for (int i = 0; i < 8; i += 4) {
    ADDSUB4H(u[0 + i], u[2 + i]);
    ADDSUB4H(u[1 + i], u[3 + i]);
    SHIFT(u[3 + i], -2);
  }
  for (int i = 0; i < 8; i += 2) { ADDSUB4H(u[i], u[i + 1]); }
  for (int i = 0; i < 8; ++i) { write4NC(u[i], out, N, line + mr * revbin[i], p); }
}

// Radix-8 DIT step. round is 0 to 3.
KERNEL(GS) void dit8d(int round, global double *in, global double *out) {
  FFT_SETUP(11, 3);
  double4 u[8];
  uint revbin[8] = {0, 4, 2, 6, 1, 5, 3, 7};

  u[0] = read4NC(in, N, line, p);
  for (int i = 1; i < 8; ++i) { u[i] = read4(in, N, line + mr * revbin[i], p + e * i); }
  for (int i = 0; i < 4; ++i) { ADDSUB4D(u[i], u[i + 4]); }
  for (int i = 1; i < 4; ++i) { SHIFT(u[i + 4], -i); }

  for (int i = 0; i < 8; i += 4) {
    ADDSUB4D(u[0 + i], u[2 + i]);
    ADDSUB4D(u[1 + i], u[3 + i]);
    SHIFT(u[3 + i], -2);
  }

  for (int i = 0; i < 8; i += 2) { ADDSUB4D(u[i], u[i + 1]); }
  for (int i = 0; i < 8; ++i) { write4NC(u[i], out, N, line + mr * revbin[i], p); }
}

void ldsWriteShifted(int u, local int *lds, int width, uint line, uint p) {
  lds[line * width + (p & (width - 1))] = (p < width) ? u : -u;
}

void ldsWriteWarped(int u, local int *lds, int width, uint line, uint p) {
  lds[line * width + ((p + line) & (width - 1))] = u;
}

/*
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
}
*/

double2 _OVL sq(double2 v) {
  double a = v.x;
  double b = v.y;
  return (double2) ((a + b) * (a - b), 2 * a * b);
}

double4 _OVL sq(double4 v) {
  double2 a = v.even;
  double2 b = v.odd;
  double2 c = a - b;
  a = sq(a);
  b = sq(b);
  c = sq(c);
  c = a + b - c;
  a = a + shift(b, 1);
  return (double4) (a.x, c.x, a.y, c.y);
}

double8 _OVL sq(double8 v) {
  double4 a = v.even;
  double4 b = v.odd;
  double4 c = a - b;
  a = sq(a);
  b = sq(b);
  c = sq(c);
  c = a + b - c;
  a = a + shift(b, 1);
  return (double8) (a.x, c.x, a.y, c.y, a.z, c.z, a.w, c.w);
}

void sq2(double *v) {
  double a = v[0];
  v[0] = (a + v[1]) * (a - v[1]);
  v[1] *= 2 * a;
}

void sq4(double *v) {
  double b[] = {v[1], v[3]};
  v[1] = v[2];
  v[2] = v[0] - b[0];
  v[3] = v[1] - b[1];
  sq2(v);
  sq2(v + 2);
  sq2(b);
  v[2] = v[0] + b[0] - v[2];
  v[3] = v[1] + b[1] - v[3];
  v[0] -= b[1];
  v[1] += b[0];
}

void sq8(double *v) {
  double b[4];
  for (int i = 0; i < 4; ++i) { b[i] = v[2 * i + 1]; } // v.odd
  for (int i = 1; i < 4; ++i) { v[i] = v[2 * i]; }     // v.even
  for (int i = 0; i < 4; ++i) { v[i + 4] = v[i] - b[i]; }
  sq4(v);
  sq4(v + 4);
  sq4(b);
  for (int i = 0; i < 4; ++i) { v[i + 4] = v[i] + b[i] - v[i + 4]; }
  v[0] -= b[3];
  v[1] += b[0];
  v[2] += b[1];
  v[3] += b[2];
}

void swapLDS(local double *lds, double *save, uint gs, uint me) {
  // Swap LDS and 'save'.
  for (int i = 0; i < 8; ++i) {
    double x = lds[i * gs + me];
    lds[i * gs + me] = save[i];
    save[i] = x;
  }
}

// Helper for sq64(). "save" is unchanged on return.
void half64(uint me, local double *lds, double *save) {
  // FFT(8), DIF 3 rounds, wrong (transposed) LDS access pattern.
  for (int round = 2; round >= 0; --round) {
    for (int i = 0; i < 4; ++i) {
      FFT_SETUP_CORE(3, round, 1, i);
      double a = read(lds, 3, me, line);
      double b = read(lds, 3, me, line + mr);
      write( a + b, lds, 3, me, line);
      writeT(a - b, lds, 3, me + e, line + mr);
    }
  }

  swapLDS(lds, save, 8, me);
  sq8(save);
  swapLDS(lds, save, 8, me);

  for (int round = 0; round < 3; ++round) {
    for (int i = 0; i < 4; ++i) {
      FFT_SETUP_CORE(3, round, 1, i);
      double a = read( lds, 3, me, line);
      double b = readT(lds, 3, me + e, line + mr);
      write(a + b, lds, 3, me, line);
      write(a - b, lds, 3, me, line + mr);
    }
  }
}

// 8 threads: 'me' in [0, 7].
// LDS is 8x8; FFT lines are 'vertical' (the 'wrong' layout).
void sq64(uint me, local double *lds) {
  double save[8];
  // Save LDS to 8 VREGs/lane.
  for (int i = 0; i < 8; ++i) {
    int p = i - me;
    double x = lds[(p & 7) * 8 + me];
    save[i] = p >= 0 ? x : -x;
  }

  half64(me, lds, save);
  swapLDS(lds, save, 8, me);
  half64(me, lds, save);

  {
    double x = save[7];
    for (int i = 7; i >= 0; --i) { save[i] += save[i - 1]; }
    save[0] -= x;
  }
  
  for (int i = 0; i < 8; ++i) {
    double x = readT(lds, 3, i + me, me);
    save[i]           += x;
    save[(i + 1) & 7] += (i < 7) ? -x : x;
  }

  for (int i = 0; i < 8; ++i) {
    write(save[i], lds, 3, i, me);
  }
}

void half2k(local double *lds, uint me) {
  for (int round = 4; round >= 0; --round) {
    for (int i = 0; i < 4; ++i) {
      FFT_SETUP_CORE(6, round, 1, me / 64 + i * 4);
      uint p = me & 63;
      double a = read(lds, 6, line, p);
      double b = read(lds, 6, line + mr, p);
      write( a + b, lds, 6, line, p);
      writeC(a - b, lds, 6, line + mr, p + e);
    }
  }

  sq64(me & 7, lds + me / 8 * 64);

  for (int round = 0; round < 5; ++round) {
    for (int i = 0; i < 4; ++i) {
      FFT_SETUP_CORE(6, round, 1, me / 64 + i * 4);
      uint p = me & 63;
      double a = read( lds, 6, line, p);
      double b = readC(lds, 6, line + mr, p + e);
      write(a + b, lds, 6, line, p);
      write(a - b, lds, 6, line + mr, p);
    }
  }
}

KERNEL(256) void sq2k(global double *tab) {
  local double lds[2048];
  uint me = get_local_id(0);
  global double *base = tab + get_group_id(0) * 2048;
  
  for (int i = 0; i < 8; ++i) {
    write(base[cut8(i * 256 + me)], lds, 6, me & 31, me / 32 + i * 8);
  }

  double save[8];
  for (int i = 0; i < 8; ++i) {
    int line = i * 4 + me / 64;
    int p    = me & 63 - 2 * line;
    double x = read(lds, 6, line, p & 63);
    save[i] = (p >= 0) ? x : -x;
  }
  
  half2k(lds, me);
  
  swapLDS(lds, save, 256, me);
  
  half2k(lds, me);
  
  for (int i = 0; i < 8; ++i) {
    int line = i * 4 + me / 64;
    int p = me & 63;
    double x = readC(lds, 6, line, p + 2 * line + 1);
    writeC(save[i] - x, lds, 6, line, p + 1);
    x = read(lds, 6, line, p);
    base[cut8(i * 256 + me)] = save[i] + x;
    /*
    write(save[i] + x, lds, 6, line, p);
    x = save[i] - x;
    lds[line * 64 + ((p + 1) & 63)] += ((p + 1) & 64) ? -x : x;
    */
  }
}


/*
void sq32(double *v) {
  double8 u[8];
  for (int i = 0; i < 4; ++i) {
    u[i] = (double8) (v[i], v[i+4], v[i+8], v[i+12], v[i+16], v[i+20], v[i+24], v[i+28]);    
  }
  u[4] = u[0];
  for (int i = 1; i < 4; ++i) { u[i + 4] = shift(u[i], i); }
  ADDSUB(u[0], u[2]);
  ADDSUB(u[1], u[3]);
  ADDSUB(u[4], u[6]);
  ADDSUB(u[5], u[7]);
  SHIFT(u[3], 2);
  SHIFT(u[7], 2);
  for (int i = 0; i < 8; i += 2) { ADDSUB(u[i], u[i+1]); }
  for (int i = 0; i < 8; ++i) { u[i] = sq(u[i]); }
  for (int i = 0; i < 8; i += 2) { ADDSUB(u[i], u[i+1]); }
  SHIFT(u[3], -2);
  SHIFT(u[7], -2);
  ADDSUB(u[0], u[2]);
  ADDSUB(u[1], u[3]);
  ADDSUB(u[4], u[6]);
  // ADDSUB(u[5], u[7]);
  u[5] = u[5] + u[7];
  SHIFT(U[5], -1);
  SHIFT(U[6], -2);
  for (int i = 0; i < 3; ++i) { ADDSUB(u[i], u[i+4]); }
  for (int i = 0; i < 3; ++i) { u[i] += shift(u[i+4], 1); }
  //todo: write to v
}
*/

/*
KERNEL(256) void sq1k(global double *io, global double *sideBase) {
  local double lds[1024];
  global double *side = sizeBase + get_group_id(0) * 2048;
  
  // Read from input, write to LDS warped.
  for (int i = 0; i < 4; ++i) {
    uint line = get_local_id(0) & 31;
    uint p = get_local_id(0) / 32 + i * 8;
    writeC(io[cut8(get_local_id(0) + i * 256)], lds, 5, line, p + line);
  }
  
  // Read from LDS warped, write to side-store and to LDS.
  for (int i = 0; i < 4; ++i) {
    bar();
    uint line = get_local_id(0) / 32;
    uint p = get_local_id(0) & 31;
    double x = readC(lds, 5, line, p + line);
    side[cut8(get_local_id(0) + i * 256)] = x;
    write(x, lds, 5, line, p);
  }

  // 5 rounds of radix-2 DIF FFT(32).
  for (int round = 4; round >= 0; --round) {
    bar();
    uint p = get_local_id(0) & 31;
    for (int i = 0; i < 2; ++i) {
      FFT_SETUP_CORE(5, round, 1, get_local_id(0) / 32 + i * 8);
      double a = read(lds, 5, line, p);
      double b = read(lds, 5, line + mr, p);
      write( a + b, lds, 5, line, p);
      writeC(a - b, lds, 5, line, p + e);
    }
  }

  // LDS transpose each block of 32 into 4 lines of 8, and save to side storage.
  bar();
  for (int i = 0; i < 4; ++i) {
    uint g = get_local_id(0) / 32;
    uint ing = get_local_id(0) & 31;
    double x = lds[i * 256 + g * 32 + (ing & 7) * 4 + ing / 8];
    side[cut8(1024 + i * 256 + get_local_id(0))] = x;
    lds[i * 256 + get_local_id(0)] = x;
  }

  // DIF FFT(4) inside blocks of 32.
  {
    bar();
    uint p = get_local_id(0) & 7;
    uint line = get_local_id(0) / 8 * 4;
    double a = read(lds, 3, line, p);
    double b = read(lds, 3, line + 2, p);
    write(a + b, lds, 3, line,     p);
    write(a - b, lds, 3, line + 2, p);
    a = read(lds, 3, line + 1, p);
    b = read(lds, 3, line + 3, p);
    write( a + b, lds, 3, line + 1, p);
    writeC(a - b, lds, 3, line + 3, p + 1);
  }
  
  // 
}
*/

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



// double2 _OVL sq(double2 v) { return (double2) (mul(v.x + v.y, v.x - v.y), mul(v.x, v.y) * 2); }

/*
double4 _OVL sq(double4 v) {
  double2 a2 = sq(v.even);
  double2 b2 = sq(v.odd);
  double2 c2 = sq(v.even - v.odd);
  double4 r;
  r.even = a2 + shift(b2, 1);
  r.odd  = a2 + b2 - c2;
  return r;
}
*/

  /*
  double4 c2 = sq(v.even - v.odd);
  double8 r;
  r.even = a2 + shift(b2, 1);
  r.odd  = a2 + b2 - c2;
  return r;

  double4 a2 = sq(v.even);
  double4 b2 = sq(v.odd);
  double4 c  = v.even - v.odd;
  v.odd  = a2 + b2;
  v.even = a2 + shift(b2, 1);
  v.odd -= sq(c);
  return v;
  */



/*
long _OVL mul(int x, int y) { return x * (long) y; }

// Negacyclic auto convolution. 2 muls
long2 _OVL sq(int2 v) { return (long2) (mul(v.x + v.y, v.x - v.y), mul(v.x, v.y) * 2); }

long4 _OVL sq(int4 v) {
  long2 a2 = sq(v.even);
  long2 b2 = sq(v.odd);
  long2 c2 = sq(v.even - v.odd);
  long4 r;
  r.even = a2 + shift(b2, 1);
  r.odd  = a2 + b2 - c2;
  return r;
}

long8 _OVL sq(int8 v) {
  long4 a2 = sq(v.even);
  long4 b2 = sq(v.odd);
  long4 c2 = sq(v.even - v.odd);
  long8 r;
  r.even = a2 + shift(b2, 1);
  r.odd  = a2 + b2 - c2;
  return r;
}
*/

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



/*
KERNEL(GS) void difIniZeropad(global int *in, global int *out) {  
  int u0 = readZeropad(in, width, j,      p);
  int u1 = readZeropad(in, width, j + mr, p);
  write(       u0 + u1, out, width, j,      p);
  writeShifted(u0 - u1, out, N, j + mr, p + e);
}

KERNEL(GS) void difIniZeropadShifted(global int *in, global int *out) {  
  int u0 = readZeropadShifted(in, width, j,      p + j);
  int u1 = readZeropadShifted(in, width, j + mr, p + j + mr);
  write(       u0 + u1, out, width, j,      p);
  writeShifted(u0 - u1, out, N, j + mr, p + e);
}
*/

/*
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
*/

/*
KERNEL(GS) void dit8try(int round, global double *io) {
  FFT_SETUP(10, 3);
  double4 u[8];
  uint revbin[8] = {0, 4, 2, 6, 1, 5, 3, 7};
  u[0] = read4NC(io, N, line, p);
  for (int i = 1; i < 8; ++i) { u[i] = read4(io, N, line + mr * revbin[i], p + e * i); }
  for (int i = 0; i < 4; ++i) { ADDSUB4D(u[i], u[i + 4]); }
  SHIFT(u[5], -1);
  SHIFT(u[6], -2);
  SHIFT(u[7], -3);
  for (int i = 0; i < 8; i += 4) {
    ADDSUB4D(u[0 + i], u[2 + i]);
    ADDSUB4D(u[1 + i], u[3 + i]);
    SHIFT(u[3 + i], -2);
  }
  for (int i = 0; i < 8; i += 2) { ADDSUB4D(u[i], u[i + 1]); }
  bar();
  for (int i = 0; i < 8; ++i) { write4NC(u[i], io, N, line + mr * revbin[i], p); }
}
*/
