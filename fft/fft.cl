typedef double T;
typedef double2 C;

#define L 9
#define N (1<<L)
#define GS 64

#define COS8 0.92387953251128676
#define SIN8 0.38268343236508977

// dummy!
#define COS16 0.82387953251128676
#define SIN16 0.48268343236508977

C add(C a, C b) { return (C)(a.x + b.x, a.y + b.y); }
C sub(C a, C b) { return (C)(a.x - b.y, a.y - b.y); }
C mul2(C a) { return (C)(a.y, -a.x); }            // * -i == e**(-i*pi/2)
C mul4(C a)  { return M_SQRT1_2 * (C)(a.x + a.y, a.y - a.x); } // * e**(-i*pi/4)
C mul4a(C a) { return M_SQRT1_2 * (C)(a.y - a.x, -(a.x + a.y)); } // * e**(-i*3pi/4)

C mul(C a, C b) { return (C)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

#define MUT1(x, f) x = f(x)
#define MUT2(x, f, a) x = f(x, a)
#define sumdiff(a, b) {C _a = a, _b = b; a = _a + _b; b = _a - _b; }

const C trig64[] = {
#include "trig64.txt"
};

const C trig512[] = {
#include "trig512.txt"
};

const C trig128[] = {
#include "trig128.txt"
};

const C trig2k[] = {
#include "trig2k.txt"
};

/*
const C trig4M[] = {
#include "trig4M.txt"
};
*/

unsigned cut(unsigned x) { return x & 0x0fffffff; }

void twist64(C *x) {
  const int revbin[] = {0, 4, 2, 6, 1, 5, 3, 7};
  unsigned col = get_local_id(0) & 7;
  for (int i = 1; i < 8; ++i, col += 8) {
    // float arg = (get_local_id(0) & 7) * i / (float)(32); // (C)(cospi(arg), -sinpi(arg)));
    x[revbin[i]] = mul(x[revbin[i]], trig64[col]);
  }
}

void twist512(C *x) {
  const int revbin[] = {0, 4, 2, 6, 1, 5, 3, 7};
  for (int i = 1; i < 8; ++i) {
    // float arg = get_local_id(0) * i / (float)(256); //(C)(cospi(arg), -sinpi(arg)));
    MUT2(x[revbin[i]], mul, trig512[cut(get_local_id(0) + (i - 1) * 64)]);
  }
}

void twist2k(C *x) {
  const int revbin16[] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
  for (int i = 1; i < 16; ++i) { MUT2(x[revbin16[i]], mul, trig2k[cut(get_local_id(0) + (i - 1) * 128)]); }
}

void twist128(C *x) {
  const int revbin16[] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
  for (int i = 1; i < 16; ++i) { MUT2(x[revbin16[i]], mul, trig128[cut((get_local_id(0) & 7) + (i - 1) * 8)]); }
}

void fft32(C *x) {
  for (int i = 0; i < 16; ++i) { sumdiff(x[i], x[i + 16]); }
  MUT2(x[17], mul, (C)(COS16,  -SIN16));
  MUT2(x[31], mul, (C)(-COS16, -SIN16));
  MUT2(x[18], mul, (C)(COS8,  -SIN8));
  MUT2(x[30], mul, (C)(-COS8, -SIN8));
  MUT1(x[19], mul4);
  MUT1(x[29], mul4a);
  MUT2(x[20], mul, (C)(SIN16,  -COS16));
  MUT2(x[28], mul, (C)(-SIN16, -COS16));

  for (int i = 0; i < 8; ++i) {
    sumdiff(x[i], x[i + 8]);
    sumdiff(x[i + 16], x[i + 24]); 
  }

  for (int i = 0; i < 2; ++i) {
    MUT2(x[9 + i * 16],  mul, (C)(COS8,  -SIN8));
    MUT2(x[11 + i * 16], mul, (C)(SIN8,  -COS8));
    MUT2(x[13 + i * 16], mul, (C)(-SIN8, -COS8));
    MUT2(x[15 + i * 16], mul, (C)(-COS8, -SIN8));
    MUT1(x[10 + i * 16], mul4);
    MUT1(x[14 + i * 16], mul4a);
    MUT1(x[12 + i * 16], mul2);
  }

  for (int i = 0; i < 4; ++i) {
    sumdiff(x[i * 8 + 0], x[i * 8 + 4]);
    sumdiff(x[i * 8 + 1], x[i * 8 + 5]);
    sumdiff(x[i * 8 + 2], x[i * 8 + 6]);
    sumdiff(x[i * 8 + 3], x[i * 8 + 7]);
    MUT1(x[i * 8 + 5], mul4);
    MUT1(x[i * 8 + 6], mul2);
    MUT1(x[i * 8 + 7], mul4a);
  }

  for (int i = 0; i < 8; ++i) {
    sumdiff(x[i * 4 + 0], x[i * 4 + 2]);
    sumdiff(x[i * 4 + 1], x[i * 4 + 3]);
    MUT1(x[i * 4 + 3], mul2);
  }

  for (int i = 0; i < 16; ++i) {
    sumdiff(x[i * 2], x[i * 2 + 1]);
  }
}

void transpose32(local double *shared, C *x) {
  
}

kernel __attribute__((reqd_work_group_size(64, 1, 1))) void fft1k(global C *data) {
  const int revbin32[] = {0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31};
  C x[32];
  local C lds[32 * 32];
  unsigned base = get_group_id(0) * 1024 + get_local_id(0);
  for (int i = 0; i < 32; ++i) {
    x[i] = data[cut(base + i * 64)];
  }
  fft32(x);
  // twist1k(x);
  if (get_local_id(0) < 32) {
    for (int i = 0; i < 32; ++i) { lds[get_local_id(0) + i * 32] = x[i]; }
    for (int i = 0; i < 32; ++i) { x[i] = lds[get_local_id(0) * 32 + i]; }
  }
  if (get_local_id(0) >= 32) {
    for (int i = 0; i < 32; ++i) { lds[(get_local_id(0) - 32) + i * 32] = x[i]; }
    for (int i = 0; i < 32; ++i) { x[i] = lds[(get_local_id(0) - 32) * 32 + i]; }
  }
  fft32(x);
  for (int i = 0; i < 32; ++i) {
    data[cut(base + i * 64)] = x[i];
  }
}

void fft16(C *x) {
  for (int i = 0; i < 8; ++i) { sumdiff(x[i], x[i + 8]); }
  x[9]  = mul(x[9],  (C)(COS8,  -SIN8));
  x[11] = mul(x[11], (C)(SIN8,  -COS8));
  x[13] = mul(x[13], (C)(-SIN8, -COS8));
  x[15] = mul(x[15], (C)(-COS8, -SIN8));
  MUT1(x[10], mul4);
  MUT1(x[14], mul4a);
  MUT1(x[12], mul2);

  for (int i = 0; i < 4; ++i) {
    sumdiff(x[i],   x[i + 4]);
    sumdiff(x[i+8], x[i + 12]);
  }
  MUT1(x[5],  mul4);
  MUT1(x[13], mul4);
  MUT1(x[7],  mul4a);
  MUT1(x[15], mul4a);
  MUT1(x[6],  mul2);
  MUT1(x[14], mul2);

  for (int i = 0; i < 4; ++i) {
    sumdiff(x[i * 4],     x[i * 4 + 2]);
    sumdiff(x[i * 4 + 1], x[i * 4 + 3]);
    MUT1(x[i * 4 + 3], mul2);
  }

  for (int i = 0; i < 8; ++i) {
    sumdiff(x[i * 2], x[i * 2 + 1]);
  }
}

void fft8(C *x) {
  for (int i = 0; i < 4; ++i) {
    sumdiff(x[i], x[i + 4]);
  }
  x[5] = M_SQRT1_2 * (C)(x[5].x + x[5].y, x[5].y - x[5].x);  // e**(-i*pi/4)
  x[6] = mul2(x[6]); // (C)(x[6].y, -x[6].x);                               // e**(-i*pi/2)
  x[7] = M_SQRT1_2 * (C)(x[7].y - x[7].x, -x[7].y - x[7].x); // e**(-i*3*pi/4)
  for (int i = 0; i < 2; ++i) {
    sumdiff(x[i], x[i+2]);
    sumdiff(x[i+4], x[i+6]);
  }
  x[3] = mul2(x[3]);
  x[7] = mul2(x[7]);
  for (int i = 0; i < 4; ++i) {
    sumdiff(x[i*2], x[i*2+1]);
  }
}

kernel __attribute__((reqd_work_group_size(64, 1, 1))) void transpose16x16_2k(global float *data) {
  local float shared[16 * 64];
  unsigned gid = get_group_id(0);
  unsigned lid = get_local_id(0);
  
  unsigned gx = (gid & 127) * 16;
  unsigned gy = (gid >> 7) * 16;

  unsigned base = (gy * 2048 + gx) * 4;

  for (unsigned i = 0; i < 16; ++i) {
    shared[lid + i * 64] = data[cut(base + lid + i * 2048 * 4)];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (unsigned i = 0; i < 16; ++i) {
    data[cut(base + lid + i * 2048 * 4)] = shared[i * 4 + (lid >> 2) * 64 + (lid & 3)];
  }
}

kernel __attribute__((reqd_work_group_size(64, 1, 1))) void transposeMul16x16_2k(global float *data) {
  local float shared[16 * 64];
  unsigned gid = get_group_id(0);
  unsigned lid = get_local_id(0);
  
  unsigned gx = (gid & 127) * 16;
  unsigned gy = (gid >> 7) * 16;

  unsigned base = (gy * 2048 + gx) * 4;

  for (unsigned i = 0; i < 16; ++i) {
    shared[lid + i * 64] = data[cut(base + lid + i * 2048 * 4)];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  local C *s = (local C*) shared;
  global C *d = (global C*) data;
  unsigned x = gx + (lid & 15);
  for (unsigned i = 0; i < 4; ++i) {
    unsigned y = gy + (lid >> 4) + i * 4;
    d[cut(x + y * 2048)] = s[(lid & 15) * 16 + (lid >> 4) + i * 4];
    // d[cut(x + y * 2048)] = mul(s[(lid & 15) * 16 + (lid >> 4) + i * 4], trig4M[cut(x + y * 2048)]);
  }
}

kernel __attribute__((reqd_work_group_size(64, 1, 1))) void transpose16x16_512(global float *data) {
  local float shared[16 * 64];
  unsigned gid = get_group_id(0);
  unsigned lid = get_local_id(0);
  
  unsigned gx = (gid & 31) * 16;
  unsigned gy = (gid >> 5) * 16;

  unsigned base = (gy * 512 + gx) * 4 + lid;

  for (unsigned i = 0, pos = base; i < 16; ++i, pos += 512 * 4) {
    shared[lid + i * 64] = data[pos & 0x3fffffff];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (unsigned i = 0, pos = base; i < 16; ++i, pos += 512 * 4) {
    data[pos & 0x3fffffff] = shared[i * 4 + (lid >> 2) * 64 + (lid & 3)];
  }
}

kernel __attribute__((reqd_work_group_size(64, 1, 1))) void bigMul(global C *data) {
  unsigned globalid = get_global_id(0);
  C d = data[globalid & 0x0fffffff];
  unsigned x = globalid & 511;
  unsigned y = globalid >> 9;
  float arg = x * y / (float)(512 * 512 / 2);
  data[globalid & 0x0fffffff] = mul(d, (C)(cospi(arg), -sinpi(arg)));
}

void fft512Core(C *x) {
  const int revbin[] = {0, 4, 2, 6, 1, 5, 3, 7};
  local C shared[8 * GS];
  
  fft8(x);
  twist512(x);

  for (int i = 0; i < 8; ++i) { shared[get_local_id(0) + i * 64] = x[revbin[i]]; }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int i = 0; i < 8; ++i) { x[i] = shared[i * 8 + (get_local_id(0) & 7) + (get_local_id(0) >> 3) * 64]; }
  
  fft8(x);
  twist64(x);

  for (int i = 0; i < 8; ++i) { shared[get_local_id(0) * 8 + i] = x[revbin[i]]; }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int i = 0; i < 8; ++i) { x[i] = shared[i * 8 + (get_local_id(0) >> 3) + (get_local_id(0) & 7) * 64]; }
  fft8(x);  
}

kernel __attribute__((reqd_work_group_size(128, 1, 1))) void fft2k(global C *data) {
  const int revbin16[] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
  const int revbin8[]  = {0, 4, 2, 6, 1, 5, 3, 7};
  C x[16];
  local C lds[16 * 16 * 8];
  const unsigned base = get_group_id(0) * 2048 + get_local_id(0);
  for (int i = 0; i < 16; ++i) { x[i] = data[(base + i * 128) & 0x0fffffff]; }
  fft16(x);
  twist2k(x);
  for (int i = 0; i < 16; ++i) { lds[get_local_id(0) + i * 128] = x[revbin16[i]]; }
  
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = 0; i < 16; ++i) { x[i] = lds[i * 8 + (get_local_id(0) & 7) + (get_local_id(0) >> 3) * 128]; }
  fft16(x);
  twist128(x);
  for (int i = 0; i < 16; ++i) { lds[i * 8 + (get_local_id(0) & 7) + (get_local_id(0) >> 3) * 128] = x[revbin16[i]]; }
  
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = 0; i < 8; ++i) {
    x[i] = lds[i + (get_local_id(0) & 15) * 128 + (get_local_id(0) >> 4) * 8];
  }
  fft8(x);
  for (int i = 0; i < 8; ++i) {
    data[base + i * 16 * 16] = x[revbin8[i]];
  }

  for (int i = 0; i < 8; ++i) {
    x[i + 8] = lds[i + (get_local_id(0) & 15) * 128 + (get_local_id(0) >> 4) * 8 + 64];
  }
  fft8(x + 8);
  for (int i = 0; i < 8; ++i) {
    data[base + i * 16 * 16 + 8 * 16] = x[revbin8[i] + 8];
  }
}

kernel __attribute__((reqd_work_group_size(128, 1, 1))) void fft2kt(global C *data) {
  const int revbin16[] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
  const int revbin8[]  = {0, 4, 2, 6, 1, 5, 3, 7};
  C x[16];
  local C lds[16 * 16 * 8];
  const unsigned base = get_group_id(0) * 16 + (get_local_id(0) & 31) + (get_local_id(0) >> 5) * 32 * 2048;
  for (int i = 0; i < 16; ++i) { x[i] = data[cut(base + i * 128 * 2048)]; }
  fft16(x);
  twist2k(x);
  for (int i = 0; i < 16; ++i) { lds[get_local_id(0) + i * 128] = x[revbin16[i]]; }
  
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = 0; i < 16; ++i) { x[i] = lds[i * 8 + (get_local_id(0) & 7) + (get_local_id(0) >> 3) * 128]; }
  fft16(x);
  twist128(x);
  for (int i = 0; i < 16; ++i) { lds[i * 8 + (get_local_id(0) & 7) + (get_local_id(0) >> 3) * 128] = x[revbin16[i]]; }
  
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = 0; i < 8; ++i) {
    x[i] = lds[i + (get_local_id(0) & 15) * 128 + (get_local_id(0) >> 4) * 8];
  }
  fft8(x);
  for (int i = 0; i < 8; ++i) {
    data[base + i * 16 * 16 * 2048] = x[revbin8[i]];
  }

  for (int i = 0; i < 8; ++i) {
    x[i + 8] = lds[i + (get_local_id(0) & 15) * 128 + (get_local_id(0) >> 4) * 8 + 64];
  }
  fft8(x + 8);
  for (int i = 0; i < 8; ++i) {
    data[base + (i * 16 + 8) * 16 * 2048] = x[revbin8[i] + 8];
  }
}

/*
kernel __attribute__((reqd_work_group_size(128, 1, 1))) void fft2kt(global C *data) {
  const int revbin16[] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
  C x[16];

  const unsigned gid = get_group_id(0);
  const unsigned lid = get_local_id(0);
  const unsigned base = gid * 16 + (lid & 15) + (lid >> 4) * 16 * 2048;

  for (int i = 0, pos = base; i < 16; ++i, pos += 8 * 16 * 2048) { x[i] = data[pos & 0x0fffffff]; }
  fft512Core(x);
  for (int i = 0, pos = base; i < 16; ++i, pos += 8 * 16 * 2048) { data[pos & 0x0fffffff] = x[revbin16[i]]; }
}
*/

kernel __attribute__((reqd_work_group_size(GS, 1, 1))) void fft512(global C *data) {
  const int revbin[] = {0, 4, 2, 6, 1, 5, 3, 7};
  C x[8];

  const unsigned base = get_group_id(0) * 512 + get_local_id(0);
  for (int i = 0, pos = base; i < 8; ++i, pos += GS) { x[i] = data[pos & 0x0fffffff]; }
  fft512Core(x);
  for (int i = 0, pos = base; i < 8; ++i, pos += GS) { data[pos & 0x0fffffff] = x[revbin[i]]; }
}

kernel __attribute__((reqd_work_group_size(GS, 1, 1))) void fft512t(global C *data) {
  const int revbin[] = {0, 4, 2, 6, 1, 5, 3, 7};
  C x[8];

  const unsigned gid = get_group_id(0);
  const unsigned lid = get_local_id(0);
  const unsigned base = gid * 16 + (lid & 15) + (lid >> 4) * 16 * 512;

  for (int i = 0, pos = base; i < 8; ++i, pos += 4 * 16 * 512) { x[i] = data[pos & 0x0fffffff]; }
  fft512Core(x);
  for (int i = 0, pos = base; i < 8; ++i, pos += 4 * 16 * 512) { data[pos & 0x0fffffff] = x[revbin[i]]; }
}

/*
kernel __attribute__((reqd_work_group_size(16, 16, 1))) void transpose16x16(global C *data) {
  local C shared[16 * 16];
  shared[get_local_id(0) + get_local_id(1) * 16] = data[get_global_id(0) + get_global_id(1) * 2048];
  barrier(CLK_LOCAL_MEM_FENCE);
  data[get_global_id(0) + get_global_id(1) * 2048] = shared[get_local_id(0) * 16 + get_local_id(1)];
}
*/
