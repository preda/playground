#define GS 256
// #define W 2048
// #define N 10

unsigned T(unsigned x) { return x & 0x3fffffff; }

// round goes down from N-1 to 0.
void difStep(int N, int width, int round, global int *in, global int *out) {
  uint groupsPerLine = width / GS;
  uint g = get_group_id(0) / groupsPerLine;
  uint k = get_group_id(0) & (groupsPerLine - 1);

  uint mr = 1 << round;
  uint j = g & (mr - 1);
  uint r = (g & ~(mr - 1)) << 1;
  uint e = j << (N - round);

  uint p0 = (j + r) * width + get_local_id(0) + k * GS;
  int u0 = in[T(p0)];
  int u1 = in[T(p0 + mr * width)];
  out[T(p0)] = u0 + u1;
  u1 = u0 - u1;
  uint p1 = get_local_id(0) + k * GS + e;
  out[T((j + r + mr) * width + (p1 & (width - 1)))] = (p1 < width) ? u1 : -u1;
}

// round goes up from 0 to N-1.
void ditStep(int N, int width, int round, global int *in, global int *out) {
  uint groupsPerLine = width / GS;
  uint g = get_group_id(0) / groupsPerLine;
  uint k = get_group_id(0) & (groupsPerLine - 1);

  uint mr = 1 << round;
  uint j = g & (mr - 1);
  uint r = (g & ~(mr - 1)) << 1;
  uint e = j << (N - round);

  uint p0 = (j + r) * width + get_local_id(0) + k * GS;
  int u0 = in[T(p0)];
  uint p1 = get_local_id(0) + k * GS + e;
  int u1 = in[T((j + r + mr) * width + (p1 & (width - 1)))];
  u1 = (p1 < width) ? u1 : -u1;
  out[T(p0)]          = u0 + u1;
  out[T(p0 + mr * width)] = u0 - u1;
}

kernel __attribute__((reqd_work_group_size(GS, 1, 1)))
void dif(int round, global int *in, global int *out) {
  difStep(10, 1024, round, in, out);
}

kernel __attribute__((reqd_work_group_size(GS, 1, 1)))
void dit(int round, global int *in, global int *out) {
  ditStep(10, 1024, round, in, out);
}

int2 sumdiff(int x, int y) { return (int2) (x + y, x - y); }
long mul(int x, int y) { return x * (long) y; }

// 7 muls
long4 negaconv4(int4 v) {
  int a = v.x, b = v.y, c = v.z, d = v.w;
  int2 sac = sumdiff(a, c);
  int2 sbd = sumdiff(b, d);
  long x = mul(sac.x, sbd.y);
  long y = mul(sac.y, sbd.x);
  long bd = mul(b, d);
  long ac = mul(a, c);
  long bc = mul(b, c);
  return (long4) (mul(sac.x, sac.y) - 2 * bd, x + y, mul(sbd.x, sbd.y) + 2 * ac,
                  (y + 2 * bc) - (x - 2 * bc));
}

int4 shift(int4 a, int e) {
  switch (e) {
  case -3: return (int4) (a.w, -a.xyz);
  case -2: return (int4) (a.zw, -a.xy);    
  case -1: return (int4) (a.yzw, -a.x);
  case  1: return (int4) (-a.w, a.xyz);
  case  2: return (int4) (-a.zw, a.xy);
  case  3: return (int4) (-a.yzw, a.x);
  default: return 0;     // assert(false);
  }
}

long4 lshift(long4 a, int e) {
  switch (e) {
  case -3: return (long4) (a.w, -a.xyz);
  case -2: return (long4) (a.zw, -a.xy);    
  case -1: return (long4) (a.yzw, -a.x);
  case  1: return (long4) (-a.w, a.xyz);
  case  2: return (long4) (-a.zw, a.xy);
  case  3: return (long4) (-a.yzw, a.x);
  default: return 0;     // assert(false);
  }
}

void set(int4 *outa, int4 *outb, int4 a, int4 b) {
  *outa = a;
  *outb = b;
}

long ha(long x, long y) { return (x >> 1) + (y >> 1) + (x & 1); }

long4 halfAdd(long4 a, long4 b) {
  return (long4) (ha(a.x, b.x), ha(a.y, b.y), ha(a.z, b.z), ha(a.w, b.w));
}

void halfAddSub(long4 *a, long4 *b) {
  long4 t = *b;
  *b = halfAdd(*a, -t);
  *a = halfAdd(*a, t);
}

// 8 * 7 muls
long16 negaconv16(int4 a, int4 b, int4 c, int4 d) {
  int4 e = a;
  int4 f = shift(b, 1);
  int4 g = shift(c, 2);
  int4 h = shift(d, 3);

  set(&a, &c, a + c, a - c);
  set(&b, &d, b + d, shift(b - d, 2));
  set(&e, &g, e + g, e - g);
  set(&f, &h, f + h, shift(f - h, 2));

  set(&a, &b, a + b, a - b);
  set(&c, &d, c + d, c - d);
  set(&e, &f, e + f, e - f);
  set(&g, &h, g + h, g - h);

  long4 la = negaconv4(a);
  long4 lb = negaconv4(b);
  long4 lc = negaconv4(c);
  long4 ld = negaconv4(d);
  long4 le = negaconv4(e);
  long4 lf = negaconv4(f);
  long4 lg = negaconv4(g);
  long4 lh = negaconv4(h);

  halfAddSub(&la, &lb);
  halfAddSub(&lc, &ld);
  halfAddSub(&le, &lf);
  halfAddSub(&lg, &lh);
  
  ld = lshift(ld, -2);
  lh = lshift(lh, -2);
  halfAddSub(&la, &lc);
  halfAddSub(&lb, &ld);
  halfAddSub(&le, &lg);
  halfAddSub(&lf, &lh);

  lf = lshift(lf, -1);
  lg = lshift(lg, -2);
  // assert(ld == lshift(lh, -3));
  halfAddSub(&la, &le);
  halfAddSub(&lb, &lf);
  halfAddSub(&lc, &lg);

  la = la + lshift(le, 1);
  lb = lb + lshift(lf, 1);
  lc = lc + lshift(lg, 1);
  
  return ((long16) (la, lb, lc, ld)).s048C159D26AE37BF;
}

kernel __attribute__((reqd_work_group_size(64, 1, 1))) void negaconv256(global int *in, global long *out) {

}

// 36 muls
void negaconv8(local int *x, local long *out) {
#define M(a, b) mul(x[a], x[b])
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

kernel __attribute__((reqd_work_group_size(256, 1, 1))) void negaconv2k(global int *in, global int *out, int round) {
  // dit2(round, in, out);
}


/*
// round: 9..0
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

// round: 0..9
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
*/
