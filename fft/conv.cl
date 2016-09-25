#define GS 256
#define W 2048
#define N 10

unsigned T(unsigned x) { return x & 0x3fffffff; }

void sumdiff(int *u) {
  int tmp = u[0];
  u[0] = tmp + u[2];
  u[2] = tmp - u[2];
  tmp = u[1];
  u[1] = tmp + u[3];
  u[3] = tmp - u[3];
}

// round 9 down to 0
void dif2(int round, global int *in, global int *out) {
  uint g = get_group_id(0) >> 3;
  uint k = get_group_id(0) & 7;

  uint mr = 1 << round;
  uint j = g & (mr - 1);
  uint r = (g & ~(mr - 1)) << 1;
  uint e = j << (N - round);

  uint p0 = (j + r) * W + get_local_id(0) + k * GS;
  int u0 = in[T(p0)];
  int u1 = in[T(p0 + mr * W)];
  out[T(p0)] = u0 + u1;
  u1 = u0 - u1;
  uint p1 = get_local_id(0) + k * GS + e;
  out[T((j + r + mr) * W + (p1 & (W - 1)))] = (p1 < W) ? u1 : -u1;
}

/* if (k < 4) {
    out[(j + r + mr) * W + p] = u1;
  } else {
  }
*/

// round 0 up to 9
void dit2(int round, global int *in, global int *out) {
  uint g = get_group_id(0) >> 3;
  uint k = get_group_id(0) & 7;

  uint mr = 1 << round;
  uint j = g & (mr - 1);
  uint r = (g & ~(mr - 1)) << 1;
  uint e = j << (N - round);

  uint p0 = (j + r) * W + get_local_id(0) + k * GS;
  int u0 = in[T(p0)];
  uint p1 = get_local_id(0) + k * GS + e;
  int u1 = in[T((j + r + mr) * W + (p1 & (W - 1)))];
  u1 = (p1 < W) ? u1 : -u1;
  out[T(p0)]          = u0 + u1;
  out[T(p0 + mr * W)] = u0 - u1;
}


kernel __attribute__((reqd_work_group_size(256, 1, 1))) void dif(global int *in, global int *out, int round) {
  dif2(round, in, out);
}

kernel __attribute__((reqd_work_group_size(256, 1, 1))) void dit(global int *in, global int *out, int round) {
  dit2(round, in, out);
}

void negaconv8(local int *x, local long *out) {
#define M(a, b) (x[a] * (long) x[b])
  out[0] = M(0, 0) - M(4, 4) - 2 * (M(1, 7) + M(2, 6) + M(3, 5));
  out[1] = 2 * (M(0, 1) - M(2, 7) - M(3, 6) - M(4, 5));
  out[2] = M(1, 1) - M(5, 5) + 2 * (M(0, 2) - M(3, 7) - M(4, 6));
  out[3] = 2 * (M(0, 3) + M(1, 2) - M(4, 7) - M(5, 6));
  out[4] = M(2, 2) - M(6, 6) + 2 * (M(0, 4) + M(1, 3) - M(5, 7));
  out[5] = 2 * (M(0, 5) + M(1, 4) + M(2, 3) - M(6, 7));
  out[6] = M(3, 3) - M(7, 7) + 2 * (M(0, 6) + M(1, 5) + M(2, 4));
  out[7] = 2 * (M(0, 7) + M(1, 6) + M(2, 5) + M(3, 4));
#undef M
}

kernel __attribute__((reqd_work_group_size(64, 1, 1))) void negaconv64(global int *in, global long *out) {  
  local int lds[16 * 8 * 2];

  uint grid = get_group_id(0);
  for (int rep = 0; rep < 8; ++rep) {
    int v = in[grid * 64 * 8 + get_local_id(0)];
    lds[rep * 128 + get_local_id(0)] = v;
    uint col = get_local_id(0) & 7;
    uint p = get_local_id(0) >> 3;
    lds[rep * 128 + 64 + (get_local_id(0) & 7) + ((p + col) & 7) * 8] = (p + col < 8) ? v : -v;
  }
  for (int round = 2; round >= 0; --round) {
    
  }
  
  

  
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
  
}

kernel __attribute__((reqd_work_group_size(256, 1, 1))) void negaconv2k(global int *in, global int *out, int round) {
  dit2(round, in, out);
}






/*
void write(int *out, uint e, uint line, uint ne, uint nq, int val) {
  uint p = e * ne + GS * nq;
  uint pos = pos + get_local_id(0);
  if (ne * (W / 4) + nq * GS >= W) {
    out[line * W + (pos & (W - 1))] = (pos < W) ? val : -val;
  } else {
    out[line * W + pos] = val;
  }
}

// 1024 x 2048; N == 10
void dif4(uint N, int round, global int *in, global int *out) {
  uint mr = (1 << (N - 2 - 2 * round));
  uint j = get_group_id(0) & (mr - 1);
  uint r = (get_group_id(0) & ~(mr - 1)) << 2;
  uint e = j << (round * 3);

  for (int quarts = 0; quarts < W / GS / 2; ++quarts) {
    int u[4];
    for (int i = 0; i < 4; ++i) { u[i] = in[(j + r + i * mr) * W + get_local_id(0) + quarts * GS]; }
    sumdiff(u);
    write(out, e, j + r,      0, quarts, u[0] + u[1]);
    write(out, e, j + r + mr, 2, quarts, u[0] - u[1]);

    int v[4];
    for (int i = 0; i < 4; ++i) { v[i] = in[(j + r + i * mr) * W + get_local_id(0) + quarts * GS + W / 2]; }
    sumdiff(v);
    write(out, e, j + r,      0, quarts + W / GS / 2, v[0] + v[1]);
    write(out, e, j + r + mr, 2, quarts + W / GS / 2, v[0] - v[1]);

    write(out, e, j + r + 2 * mr, 1, quarts, u[2] - v[3]);
    write(out, e, j + r + 3 * mr, 3, quarts, u[2] + v[3]);
    
    write(out, e, j + r + 2 * mr, 1, quarts + W / GS / 2, v[2] + u[3]);
    write(out, e, j + r + 3 * mr, 3, quarts + W / GS / 2, v[2] - u[3]);
  }
}

// 1024 x 2048; N == 10
void dif4a(int round, global int *in, global int *out) {
  uint mr = 1 << (N - 2 - 2 * round);
  uint g = get_group_id(0) >> 2;
  uint j = g & (mr - 1);
  uint r = (g & ~(mr - 1)) << 2;
  uint e = j << (round * 2 + 1);

  uint quarts = get_group_id(0) & 3;
  int u[4];
  for (int i = 0; i < 4; ++i) { u[i] = in[(j + r + i * mr) * W + get_local_id(0) + quarts * GS]; }
  sumdiff(u);
  write(out, e, j + r,      0, quarts, u[0] + u[1]);
  write(out, e, j + r + mr, 2, quarts, u[0] - u[1]);
  
  int v[4];
  for (int i = 0; i < 4; ++i) { v[i] = in[(j + r + i * mr) * W + get_local_id(0) + quarts * GS + W / 2]; }
  sumdiff(v);
  write(out, e, j + r,      0, quarts + W / GS / 2, v[0] + v[1]);
  write(out, e, j + r + mr, 2, quarts + W / GS / 2, v[0] - v[1]);
  
  write(out, e, j + r + 2 * mr, 1, quarts, u[2] - v[3]);
  write(out, e, j + r + 3 * mr, 3, quarts, u[2] + v[3]);
  
  write(out, e, j + r + 2 * mr, 1, quarts + W / GS / 2, v[2] + u[3]);
  write(out, e, j + r + 3 * mr, 3, quarts + W / GS / 2, v[2] - u[3]);
}
*/
