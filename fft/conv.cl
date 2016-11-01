#define T int
#include "template.h"
#undef T

#define T long
#include "template.h"
#undef T

#define GS 256

unsigned cut(unsigned x) { return x & 0x3fffffff; }

int read(global int *in, int width, int line, int p) {
  return in[cut(line * width + p)];
}

void write(int u, global int *out, int width, int line, int p) {
  out[cut(line * width + p)] = u;
}

int readShifted(global int *in, int width, int line, int p) {
  int u = read(in, width, line, p & (width - 1));
  return (p < width) ? u : -u;
}

void writeShifted(int u, gloabl int *out, int width, int line, int p) {
  write((p < width) ? u : -u, out, width, line, p & (width - 1));
}

// round goes down from N-1 to 0.
void difStep(int N, int round, global int *in, global int *out) {
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

void difFirst(int N, global int *in, global int *out) {
  int width = 1 << N;
  uint groupsPerLine = width / GS;
  uint g = get_group_id(0) / groupsPerLine;
  uint k = get_group_id(0) & (groupsPerLine - 1);

  uint mr = 1 << (N - 1);
  uint j = g;
  uint e = j;
  uint p = get_local_id(0) + k * GS;
  
  int u0 = readShifted(in, width, j,      p + j);
  int u1 = readShifted(in, width, j + mr, p + j + mr);
  write(       u0 + u1, out, width, j,      p);
  writeShifted(u0 - u1, out, width, j + mr, p + e);
}

// round goes up from 0 to N-1.
void ditStep(int N, int round, global int *in, global int *out) {
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
  write(halfAdd1int(u0, u1),  out, width, j + r,      p);
  write(halfAdd1int(u0, -u1), out, width, j + r + mr, p);
}

void ditLast(int N, global int *in, global int *out) {
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
  writeShifted(halfAdd1int(u0, u1),  out, width, j,      p + j);
  writeShifted(halfAdd1int(u0, -u1), out, width, j + mr, p + mr);
}

// DIF step for a 2**12 FFT
kernel __attribute__((reqd_work_group_size(GS, 1, 1)))
void dif(int round, global int *in, global int *out) {
  difStep(12, round, in, out);
}

kernel __attribute__((reqd_work_group_size(GS, 1, 1)))
void difShiftedIn(int round, global int *in, global int *out) {
  difFirst(12, in, out);
}

// DIT step for a 2**12 FFT
kernel __attribute__((reqd_work_group_size(GS, 1, 1)))
void dit(int round, global int *in, global int *out) {
  ditStep(12, round, in, out);
}

kernel __attribute__((reqd_work_group_size(GS, 1, 1)))
void ditShiftedOut(int round, global int *in, global int *out) {
  ditLast(12, in, out);
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
  return ((long4) (x2 + shift2l(y2), x2 + y2 - sq2(v.x - v.y))).xzyw;
}

// 18 muls (3 x sq4())
long8 sq8(int8 v) {
  int4 x = v.s0246;
  int4 y = v.s1357;
  long4 x2 = sq4(x);
  long4 y2 = sq4(y);
  return ((long8) (x2 + shift(y2, 1), x2 + y2 - sq4(x - y))).xzyw;
}

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
  for (uint round = 2; round >= 0; --round) {
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
  return (long4) (halfAdd1long(a.x, b.x), halfAdd1long(a.y, b.y), halfAdd1long(a.z, b.z), halfAdd1long(a.w, b.w));
}

void halfAddSub(long4 *a, long4 *b) {
  long4 t = *b;
  *b = halfAdd4(*a, -t);
  *a = halfAdd4(*a, t);
}



// 8 * 6 muls
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
