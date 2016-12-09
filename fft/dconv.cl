#define _O __attribute__((overloadable))
#define GS 256
#define KERNEL kernel __attribute__((reqd_work_group_size(GS, 1, 1)))

double4 _O shift(double4 a, int e) {
  switch (e) {
  case -3: return (double4) (a.w, -a.xyz);
  case -2: return (double4) (a.zw, -a.xy);
  case -1: return (double4) (a.yzw, -a.x);
  case  1: return (double4) (-a.w, a.xyz);
  case  2: return (double4) (-a.zw, a.xy);
  case  3: return (double4) (-a.yzw, a.x);
  default: return 0;
  }
}

void bar() { barrier(CLK_LOCAL_MEM_FENCE); }
unsigned cut8(unsigned x) { return x & 0x1fffffff; }

double _O read(global double *in, uint W, uint line, uint p) { return in[cut8(line * W + p)]; }
void _O write(double u, global double *out, uint W, uint line, uint p) { out[cut8(line * W + p)] = u; }

double _O readC(global double *in, uint W, uint line, uint p) {
  double u = read(in, W, line, p % W);
  return (p & W) ? -u : u;
}

void _O writeC(double u, global double *out, uint W, uint line, uint p) {
  write((p & W) ? -u : u, out, W, line, p % W);
}

double4 _O read4(global double *in, uint W, uint line, uint p) {
  double u[4];
  for (int i = 0; i < 4; ++i) { u[i] = readC(in, W, line, p + (W / 4) * i); }
  return (double4)(u[0], u[1], u[2], u[3]);
}

double4 _O read4NC(global double *in, uint W, uint line, uint p) {
  double u[4];
  for (int i = 0; i < 4; ++i) { u[i] = read(in, W, line, p + (W / 4) * i); }
  return (double4)(u[0], u[1], u[2], u[3]);
}

void _O write4NC(double4 u, global double *out, uint W, uint line, uint p) {
  for (int i = 0; i < 4; ++i) { write((double[4]){u.x, u.y, u.z, u.w}[i], out, W, line, p + (W / 4) * i); }
}

#define ADDSUB(a, b)  { double  tmp = a; a = tmp + b; b = tmp - b; }
#define ADDSUB4(a, b) { double4 tmp = a; a = tmp + b; b = tmp - b; }
// #define ADDSUB4(a, b) { double4 tmp = b; b = a - b; a = a + tmp; }
#define SHIFT(u, e) u = shift(u, e);

void fft(bool isDIF, const uint W, const uint round, global double *in, global double *out) {
  uint groupsPerLine = W / 4 / GS;
  uint g = get_group_id(0) / groupsPerLine;
  uint mr = 1 << round;
  uint j = g & (mr - 1);
  uint r = (g & ~(mr - 1)) * 8;
  uint e = (j * (W / 4)) >> round;
  uint line = j + r;
  uint k = get_group_id(0) % groupsPerLine;
  uint p = get_local_id(0) + k * GS;

  double4 u[8];
  uint revbin[8] = {0, 4, 2, 6, 1, 5, 3, 7};

  if (isDIF) {
    // DIF
    for (int i = 0; i < 8; ++i) { u[i] = read4NC(in, W, line + mr * i, p); }
    for (int i = 0; i < 4; ++i) { ADDSUB4(u[i], u[i + 4]); }
    for (int i = 1; i < 4; ++i) { SHIFT(u[i + 4], i); }

    for (int i = 0; i < 8; i += 4) {
      ADDSUB4(u[0 + i], u[2 + i]);
      ADDSUB4(u[1 + i], u[3 + i]);
      SHIFT(u[3 + i], 2);
    }

    for (int i = 0; i < 8; i += 2) { ADDSUB4(u[i], u[i + 1]); }
    write4NC(u[0], out, W, line, p);
    for (int i = 1; i < 8; ++i) {
      for (int q = 0; q < 4; ++q) {
        double x = (double[4]){u[i].x, u[i].y, u[i].z, u[i].w}[q];
        // if ((int) (k + 1) * (int) GS + (int) e * (int) revbin[i] - (4 - q) * (int) (W / 4) <= 0) {
        if ((k + 1) * GS + e * revbin[i] - (4 - q) * (W / 4) <= 0) {
          write(x, out, W, line + mr * i, p + e * revbin[i] + q * (W / 4));
        } else {
          writeC(x, out, W, line + mr * i, p + e * revbin[i] + q * (W / 4));
        }
      }
    }
    
  } else {
    // DIT
    u[0] = read4NC(in, W, line, p);
    for (int i = 1; i < 8; ++i) { u[i] = read4(in, W, line + mr * revbin[i], p + e * i); }
    for (int i = 0; i < 4; ++i) { ADDSUB4(u[i], u[i + 4]); }
    for (int i = 1; i < 4; ++i) { SHIFT(u[i + 4], -i); }

    for (int i = 0; i < 8; i += 4) {
      ADDSUB4(u[0 + i], u[2 + i]);
      ADDSUB4(u[1 + i], u[3 + i]);
      SHIFT(u[3 + i], -2);
    }

    for (int i = 0; i < 8; i += 2) { ADDSUB4(u[i], u[i + 1]); }
    for (int i = 0; i < 8; ++i) { write4NC(u[i], out, W, line + mr * revbin[i], p); }
  }
}

void difStep(const uint W, const uint round, global double *in, global double *out) {
  fft(true, W, round, in, out);
}

void ditStep(const uint W, const uint round, global double *in, global double *out) {
  fft(false, W, round, in, out);
}

#define W 4096
KERNEL void dif(uint round, global double *in, global double *out) { difStep(W, round, in, out); }
KERNEL void dit(uint round, global double *in, global double *out) { ditStep(W, round, in, out); }
KERNEL void dif_0(global double *in, global double *out) { difStep(W, 0, in, out); }
#undef W

void conv4kAux(local double *lds) {
  uint me = get_local_id(0);
  uint p = me % 64;

  for (int round = 5; round >= 0; --round) {
    bar();
    uint mr = 1 << round;
    for (int i = 0; i < 8; ++i) {
      uint g = me / 64 + i * 4;
      uint j = g & (mr - 1);
      uint r = (g & ~(mr - 1)) * 2;
      uint e = j * (64 >> round);
      uint line = j + r;
      double a = lds[line * 64 + p];
      double b = lds[(line + mr) * 64 + p];
      ADDSUB(a, b);
      lds[line * 64 + p] = a;
      // bar();
      lds[(line + mr) * 64 + (p + e) % 64] = ((p + e) & 64) ? -b : b;
    }
  }

  for (int round = 0; round < 6; ++round) {
    bar();
    uint mr = 1 << round;
    for (int i = 0; i < 8; ++i) {
      uint g = me / 64 + i * 4;
      uint j = g & (mr - 1);
      uint r = (g & ~(mr - 1)) * 2;
      uint e = j * (64 >> round);
      uint line = j + r;
      double a = lds[line * 64 + p];
      double b = lds[(line + mr) * 64 + (p + e) % 64];
      b = ((p + e) & 64) ? -b : b;
      ADDSUB(a, b);
      lds[line * 64 + p] = a;
      // bar();
      lds[(line + mr) * 64 + p] = b;
    }
  }
}

KERNEL void conv4k(global double *in, global double *out) {
  local double lds[4096]; // 32 KB
  double u[16];
  
  in  += get_group_id(0) * 4096;
  out += get_group_id(0) * (4096 * 2);

  uint me = get_local_id(0);
  uint p = me % 64;
  
  for (int i = 0; i < 16; ++i) { 
    lds[i * 4 + me / 64 + p * 64] = u[i] = in[cut8(me + i * 256)];
  }

  conv4kAux(lds);

  bar();  
  for (int i = 0; i < 16; ++i) {
    double tmp = lds[i * 4 + me / 64 + p * 64];
    bar();
    lds[i * 4 + me / 64 + p * 64] = u[i];
    u[i] = tmp;
  }

  // conv4kAux(lds);
  
  /*
  out += 4096;
  bar();
  for (int i = 0; i < 16; ++i) {
    lds[i * 4 + me / 64 + p * 64] = u[i];
      // in[cut8(me + i * 256)];
  }
  */

  bar();
  for (int i = 0; i < 16; ++i) {
    out[cut8(me + i * 256)] = lds[i * 4 + me / 64 + p * 64];
  }
}

KERNEL void round0(global double *in, global double *out) {
  uint g = get_group_id(0);
  uint line = g / 8 * 2;
  uint p = line * 2048 + get_local_id(0) + (g % 8) * 256;
  double a = in[p];
  double b = in[p + 2048];
  out[p] = a + a * b;
  out[p + 2048] = a - a * b;
}

KERNEL void copy(global double *in, global double *out) {
  uint g = get_group_id(0);
  out[g * 256 + get_local_id(0)] = in[g * 256 + (get_local_id(0) + 1) % 256];
  // (get_local_id(0) + 1) % 256
}


/*
KERNEL void dif_3(global double *in, global double *out) { difStep(W, 3, in, out); }
KERNEL void dif_6(global double *in, global double *out) { difStep(W, 6, in, out); }
KERNEL void dit_0(global double *in, global double *out) { ditStep(W, 0, in, out); }
KERNEL void dit_3(global double *in, global double *out) { ditStep(W, 3, in, out); }
KERNEL void dit_6(global double *in, global double *out) { ditStep(W, 6, in, out); }
*/

