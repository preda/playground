
T##2 shift2##T(T##2 a, int e) {
  switch (e) {
  case -1: return (T##2) (a.y, -a.x);
  case  1: return (T##2) (-a.y, a.x);
  default: return 0; // assert(false);
  }
}

T##4 shift4##T(T##4 a, int e) {
  switch (e) {
  case -3: return (T##4) (a.w, -a.xyz);
  case -2: return (T##4) (a.zw, -a.xy);    
  case -1: return (T##4) (a.yzw, -a.x);
  case  1: return (T##4) (-a.w, a.xyz);
  case  2: return (T##4) (-a.zw, a.xy);
  case  3: return (T##4) (-a.yzw, a.x);
  default: return 0; // assert(false);
  }
}

T halfAdd1(T x, T y) { return (x >> 1) + (y >> 1) + (x & 1); }


/*
int2 shift2(int2 a, int e) {
  switch (e) {
  case -1: return (int2) (a.y, -a.x);
  case  1: return (int2) (-a.y, a.x);
  }
}

long2 shift2l(long2 a, int e) {
  switch (e) {
  case -1: return (long2) (a.y, -a.x);
  case  1: return (long2) (-a.y, a.x);
  }
}

int4 shift4(int4 a, int e) {
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

long4 shift4l(long4 a, int e) {
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

long ha(long x, long y) { return (x >> 1) + (y >> 1) + (x & 1); }
int iha(int x, int y) { return (x >> 1) + (y >> 1) + (x & 1); }

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

// 7 muls
/*
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
*/


*/
