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

double _O read(local double *lds, uint W, uint line, uint p) { return lds[line * W + p]; }
void _O write(double u, local double *lds, uint W, uint line, uint p) { lds[line * W + p] = u; }

double _O readR(local double *lds, uint W, uint line, uint p) { return lds[line * W + p % W]; }
double _O readC(local double *lds, uint W, uint line, uint p) {
  double u = lds[line * W + p % W];
  return (p & W) ? -u : u;
}

void _O writeR(double u, local double *lds, uint W, uint line, uint p) { lds[line * W + p % W] = u; }
void _O writeC(double u, local double *lds, uint W, uint line, uint p) { lds[line * W + p % W] = (p & W) ? -u : u; }

double2 read2(local double *lds, uint W, uint line, uint p) {
  return (double2){lds[line * W + p], lds[line * W + p + W / 2]};
}

double2 read2C(local double *lds, uint W, uint line, uint p) {
  return (double2){readC(lds, W, line, p), readC(lds, W, line, p + W / 2)};
}

void write2(double2 u, local double *lds, uint W, uint line, uint p) {
  write(u.x, lds, W, line, p);
  write(u.y, lds, W, line, p + W / 2);
}

void write2C(double2 u, local double *lds, uint W, uint line, uint p) {
  writeC(u.x, lds, W, line, p);
  writeC(u.y, lds, W, line, p + W / 2);
}

#define ADDSUB(a, b)  { double  tmp = a; a = tmp + b; b = tmp - b; }
#define ADDSUB2(a, b) { ADDSUB(a.x, b.x); ADDSUB(a.y, b.y); }
#define ADDSUB4(a, b) { double4 tmp = a; a = tmp + b; b = tmp - b; }
#define SHIFT(u, e) u = shift(u, e);

#define SWAP(a, b) { double t = a.x; a.x = b.x; b.x = t; t = a.y; a.y = b.y; b.y = t; }

double2 _O mul(double2 u, double a, double b) { return (double2) { u.x * a - u.y * b, u.x * b + u.y * a}; }
// double2 mulm1(double2 u, double a) { return (double2) { u.x * a + u.y, u.y * a - u.x}; }

// cos(pi / 8), sin(pi / 8)
#define C8 0.92387953251128676
#define S8 0.38268343236508977


void fft16(double2 *r) {
  for (int i = 0; i < 8; ++i) { ADDSUB2(r[i], r[i + 8]); }
  r[9]  = mul(r[9], C8, -S8);
  r[10] = mul(r[10], 1, -1) * M_SQRT1_2;
  r[11] = mul(r[11], S8, -C8);
  r[12] = mul(r[12], 0, -1);
  r[13] = mul(r[13], -S8, -C8);
  r[14] = mul(r[14], -1, -1) * M_SQRT1_2;
  r[15] = mul(r[15], -C8, -S8);

  for (int i = 0; i < 4; ++i) {
    ADDSUB2(r[i],     r[i + 4]);
    ADDSUB2(r[i + 8], r[i + 12]);
  }
  
  r[5]  = mul(r[5],   1, -1) * M_SQRT1_2;
  r[6]  = mul(r[6],   0, -1);
  r[7]  = mul(r[7],  -1, -1) * M_SQRT1_2;
  r[13] = mul(r[13],  1, -1) * M_SQRT1_2;
  r[14] = mul(r[14],  0, -1);
  r[15] = mul(r[15], -1, -1) * M_SQRT1_2;

  for (int i = 0; i < 4; ++i) {
    ADDSUB2(r[i * 4],     r[i * 4 + 2]);
    ADDSUB2(r[i * 4 + 1], r[i * 4 + 3]);
    r[i * 4 + 3] = mul(r[i * 4 + 3], 0, -1);
  }

  for (int i = 0; i < 8; ++i) {
    ADDSUB2(r[i * 2], r[i * 2 + 1]);
  }
  
  // revbin(16)
  SWAP(r[1],  r[8]);
  SWAP(r[2],  r[4]);
  SWAP(r[3],  r[12]);
  SWAP(r[5],  r[10]);
  SWAP(r[7],  r[14]);
  SWAP(r[11], r[13]);
}

KERNEL void convfft(global double *buf) {
  buf += get_group_id(0) * 4096;
  
  local double lds[4096];

  uint me = get_local_id(0);

  double2 r[16];
  
  for (int i = 0; i < 16; ++i) { r[i] = (double2){buf[i * 256 + me], 0}; }
  fft16(r);

  

  
  
  for (int i = 0; i < 16; ++i) { buf[i * 256 + me] = r[i].x; }
}


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
  uint p = me % 32;

  for (int round = 4; round >= 2; round -= 2) {
    bar();
    uint mr = 1 << round;
    #pragma unroll 1
    for (int i = 0; i < 2; ++i) {
      uint g = me / 32 + i * 8;
      uint j = g & (mr - 1);
      uint r = (g & ~(mr - 1)) * 4;
      uint e = j * (32 >> round);
      uint line = j + r;

      double2 u0 = read2(lds, 64, line + mr * 0, p);
      double2 u1 = read2(lds, 64, line + mr * 1, p);
      double2 u2 = read2(lds, 64, line + mr * 2, p);
      double2 u3 = read2(lds, 64, line + mr * 3, p);
      ADDSUB2(u0, u2);
      ADDSUB2(u1, u3);
      u3 = (double2) {-u3.y, u3.x}; //shift(u3, 1);
      ADDSUB2(u0, u1);
      ADDSUB2(u2, u3);
      write2C(u0, lds, 64, line, p);
      write2C(u1, lds, 64, line + mr, p + e * 2);
      write2C(u2, lds, 64, line + mr * 2, p + e);
      write2C(u3, lds, 64, line + mr * 3, p + e * 3);
    }
  }

  for (int i = 0; i < 2; ++i) {
    uint line  = i * 32 + (me / 32) * 4;
    double2 u0 = read2(lds, 64, line + 0, p);
    double2 u1 = read2(lds, 64, line + 1, p);
    double2 u2 = read2(lds, 64, line + 2, p);
    double2 u3 = read2(lds, 64, line + 3, p);
    ADDSUB2(u0, u2);
    ADDSUB2(u1, u3);
    u3 = (double) {-u3.y, u3.x};
    ADDSUB2(u0, u1);
    ADDSUB2(u2, u3);
    write2(u0, lds, 64, line + 0, p);
    write2(u1, lds, 64, line + 1, p);
    write2(u2, lds, 64, line + 2, p);
    write2(u3, lds, 64, line + 3, p);
  }


  /*
  bar();
  #pragma unroll 1
  for (int i = 0; i < 16; ++i) {
    double x = lds[i * 256 + me];
    lds[i * 256 + me] = x * x;
  }
  */

  /*
  bar();
  #pragma unroll 1
  for (int i = 0; i < 16; ++i) {
    double x = lds[i * 256 + me];
    uint c = me % 4;
    uint l = (me / 4) % 16;
    bar();
    lds[i * 256 + (me / 64) * 64 + c * 16 + l] = x;
  }
  */

  bar();
  #pragma unroll 1
  for (int i = 0; i < 16; ++i) {
    double x = lds[i * 256 + me];
    uint c = me % 8;
    uint l = (me / 8) % 8;
    bar();
    lds[i * 256 + (me / 64) * 64 + c * 8 + l] = x;
  }

  bar();
  for (int i = 0; i < 8; ++i) {
    uint c = me % 8;
    uint l = (me % 32) / 8;
    uint line = i * 64 + me / 32 * 8 + l;
    double a = read(lds, 8, line,     c);
    double b = read(lds, 8, line + 4, c);
    ADDSUB(a, b);
    write( a, lds, 8, line,     c);
    writeC(b, lds, 8, line + 4, c + l * 2);
  }

  bar();
  for (int i = 0; i < 2; ++i) {
    uint c = me % 4;
    uint l = (me % 8) / 4 * 4;
    uint line = i * 256 + me / 8 * 8 + l;
    double2 u0 = read2(lds, 8, line,     c);
    double2 u1 = read2(lds, 8, line + 1, c);
    double2 u2 = read2(lds, 8, line + 2, c);
    double2 u3 = read2(lds, 8, line + 3, c);
    ADDSUB2(u0, u2);
    ADDSUB2(u1, u3);
    u3 = (double) {-u3.y, u3.x};
    ADDSUB2(u0, u1);
    ADDSUB2(u2, u3);
    write2(u0, lds, 8, line, c);
    write2(u1, lds, 8, line + 1, c);
    write2(u2, lds, 8, line + 2, c);
    write2(u3, lds, 8, line + 3, c);
  }

  bar();
  for (int i = 0; i < 2; ++i) {
    double x[8];
    for (int k = 0; k < 8; ++k) {
      x[k] = read(lds, 8, i * 256 + me, k);
    }
    for (int k = 0; k < 8; ++k) {
      x[k] = x[k] * x[8 - k];
    }
    for (int k = 0; k < 8; ++k) {
      write(x[k], lds, 8, i * 256 + me, k);
    }
  }

  bar();
  for (int i = 0; i < 2; ++i) {
    uint c = me % 4;
    uint l = (me % 8) / 4 * 4;
    uint line = i * 256 + me / 8 * 8 + l;
    double2 u0 = read2(lds, 8, line,     c);
    double2 u2 = read2(lds, 8, line + 1, c);
    double2 u1 = read2(lds, 8, line + 2, c);
    double2 u3 = read2(lds, 8, line + 3, c);
    ADDSUB2(u0, u2);
    ADDSUB2(u1, u3);
    u3 = (double) {-u3.y, u3.x};
    ADDSUB2(u0, u1);
    ADDSUB2(u2, u3);
    write2(u0, lds, 8, line, c);
    write2(u2, lds, 8, line + 1, c);
    write2(u1, lds, 8, line + 2, c);
    write2(u3, lds, 8, line + 3, c);
  }
   
  for (int round = 0; round < 6; round += 2) {
    bar();
    uint mr = 1 << round;
    for (int i = 0; i < 2; ++i) {
      uint g = me / 32 + i * 8;
      uint j = g & (mr - 1);
      uint r = (g & ~(mr - 1)) * 4;
      uint e = j * (32 >> round);
      uint line = j + r;

      double2 u0 = read2( lds, 64, line + mr * 0, p);
      double2 u2 = read2C(lds, 64, line + mr * 1, p + e * 2);
      double2 u1 = read2C(lds, 64, line + mr * 2, p + e * 1);
      double2 u3 = read2C(lds, 64, line + mr * 3, p + e * 3);
      ADDSUB2(u0, u2);
      ADDSUB2(u1, u3);
      u3 = (double2) {u3.y, -u3.x}; // shift(u3, -1);
      ADDSUB2(u0, u1);
      ADDSUB2(u2, u3);
      write2(u0, lds, 64, line + mr * 0, p);
      write2(u2, lds, 64, line + mr * 1, p);
      write2(u1, lds, 64, line + mr * 2, p);
      write2(u3, lds, 64, line + mr * 3, p);
    }
  }
}

KERNEL void conv4k(global double * restrict in, global double * restrict out) {
  local double lds[4096]; // i.e. 32 KB
  
  in  += get_group_id(0) * 4096;
  out += get_group_id(0) * 4096;

  uint me = get_local_id(0);
  uint p = me % 64;

  #pragma unroll 1
  for (int i = 0; i < 16; ++i) {
    double x = in[cut8(me + i * 256)];
    writeC(x, lds, 64, p, i * 4 + me / 64 + p); // lds[p * 64 + col % 64] = (col < 64) ? x : -x;;
  }

  bar();
  conv4kAux(lds);

  /*
  bar();
  for (int i = 0; i < 16; ++i) {
    uint line = me / 64 + i * 4;
    double tmp = readC(lds, 64, line, p + line);
    tmp -= readC(lds, 64, line, p + line - 1);
    write(u[i], lds, 64, line, p);
    u[i] = tmp;
  }
  */

  bar();
  // #pragma unroll 1
  for (int i = 0; i < 16; ++i) {
    out[cut8(me + i * 256)] = readC(lds, 64, p, i * 4 + me / 64 + p);
    // lds[i * 4 + me / 64 + p * 64];
  }
}

KERNEL void conv4kbig(global double *in, global double *out) {
  local double lds[4096]; // i.e. 32 KB
  
  in  += get_group_id(0) * 4096;
  out += get_group_id(0) * 4096;

  uint me = get_local_id(0);
  uint p = me % 64;
  
  for (int i = 0; i < 16; ++i) {
    double x = in[cut8(me + i * 256)];
    writeC(x, lds, 64, p, i * 4 + me / 64 + p); // lds[p * 64 + col % 64] = (col < 64) ? x : -x;;
  }

  double u[16];
  
  bar();
  for (uint i = 0; i < 16; ++i) {
    uint line = i * 4 + me / 64;
    double x = lds[line * 64 + (p + line) % 64];
    u[i] = (p + line < 64) ? x : -x;
  }

  conv4kAux(lds);

  bar();
  #pragma unroll 1
  for (int i = 0; i < 16; ++i) {
    uint line = me / 64 + i * 4;
    double tmp = readC(lds, 64, line, p + line);
    tmp -= readC(lds, 64, line, p + line - 1);
    write(u[i], lds, 64, line, p);
    u[i] = tmp;
  }

  /*
  for (int i = 0; i < 16; ++i) {    
    double tmp = lds[i * 256 + me];
    lds[i * 256 + me] = u[i];
    u[i] = tmp;
  }
  */

  conv4kAux(lds);

  bar();
  #pragma unroll 1
  for (int i = 0; i < 16; ++i) {
    uint line = me / 64 + i * 4;
    u[i] += readC(lds, 64, line, p - 1);
    u[i] += read(lds, 64, line, p);
    write(u[i], lds, 64, line, p);
  }

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

  /*
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
    }*/
