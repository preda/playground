#include <assert.h>
#include <stdint.h>
#include <unordered_map>
#include <string>
#include <utility>
#include <immintrin.h>

using std::string;

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned u32;
typedef uint64_t u64;
typedef unsigned __int128 u128;

inline auto min(auto a, auto b) { return a < b  ? a : b; }
inline auto max(auto a, auto b) { return a >= b ? a : b; }

inline int firstOf(u64 bits) { assert(bits); return __builtin_ctzll(bits); }
inline int firstOf(u32 bits) { assert(bits); return __builtin_ctz(bits);   }
inline int size(u64 bits) { return __builtin_popcountll(bits); }
inline int size(u32 bits) { return __builtin_popcount(bits); }


inline bool IS(int p, u64 bits) { assert(p >= 0 && p < 64); return bits & (1ull << p); }
inline void SET(int p, u64 &bits) { assert(p >= 0 && p < 64); bits |= (1ull << p); }
inline void SET(int p, u32 &bits) { assert(p >= 0 && p < 32); bits |= (1 << p); }
constexpr void SETC(int p, u64 &bits) { bits |= (1ull << p); }

inline void CLEAR(int p, u64 &bits) { assert(p >= 0 && p < 64); bits &= ~(1ull << p); }
inline int  POP(u64 &bits) { int p = firstOf(bits); CLEAR(p, bits); return p; }

enum {
  SIZE = 3,
  TOTAL_POINTS = SIZE * SIZE,
  DELTA = 8,
  PASS = 63,
};

constexpr inline int P(int y, int x) { return (y << 3) + x; }
inline int Y(int pos) { return pos >> 3; }
inline int X(int pos) { return pos & 7; }

constexpr u64 insidePoints() {
  u64 ret = 0;
  for (int y = 0; y < SIZE; ++y) {
    for (int x = 0; x < SIZE; ++x) {
      SETC(P(y, x), ret);
    }
  }
  return ret;
}

#define PURE const __attribute__((warn_unused_result))
#define STR(x) ((string) x).c_str()
#define NEIB(p) {p + 1, p + DELTA, p - 1, p - DELTA}

enum {
  INSIDE = insidePoints(),
};

class Value {  
  enum Kind {AT_LEAST, AT_MOST, DEEPER, LOOP, NIL};

  static Kind negateKind(Kind kind) { return (kind == AT_MOST) ? AT_LEAST : ((kind == AT_LEAST) ? AT_MOST : kind); }

  Value(Kind kind, int v, int depth): kind(kind), v(v), depth(depth) {}
  
public:
  Kind kind;
  int v;
  int depth;

  static Value loop(int depth) { return Value(LOOP, 0, depth); }
  static Value nil()    { return Value(NIL, 0, 0); }
  static Value deeper() { return Value(DEEPER, 0, 0); }
  static Value atMost(int k)  { return Value(AT_MOST, k, 0); }
  static Value atLeast(int k) { return Value(AT_LEAST, k, 0); }

  operator string() {
    char buf[64];
    snprintf(buf, sizeof(buf), "V(%d, %d, %d)", kind, v, depth);
    return buf;
  }

  bool operator==(const Value &o) { return kind == o.kind && v == o.v && depth == o.depth; }
  
  bool isAtLeast(int k) PURE { return kind == AT_LEAST && v >= k; }
  bool isAtMost(int k)  PURE { return kind == AT_MOST  && v <= k; }
  bool isDeeper()       PURE { return kind == DEEPER; }
  bool isLoop()         PURE { return kind == LOOP; }
  bool isNil()          PURE { return kind == NIL; }
  
  bool isFinalEnough(int k) PURE { return isDeeper() || isAtLeast(k + 1) || isAtMost(k) || isLoop(); }
  bool isCut(int k) PURE { return isAtLeast(k + 1); }

  Value negate() PURE { return Value(negateKind(kind), -v, depth); }

  void accumulate(const Value &sub, int k) {
    assert(!isCut(k));
    if (sub.isCut(k) || sub.isDeeper() || isNil()) {
      *this = sub;
    } else if (!isDeeper()) {
      if (isLoop() || sub.isLoop()) {
        *this = loop(min(depth, sub.depth));
      } else {
        assert(sub.kind == AT_MOST && kind == AT_MOST);
        v = max(v, sub.v);  // atMost(max(v, sub.v));
      }
    }
  }

  Value fromTT(int k, int d) PURE {
    return
      (kind == DEEPER && v == k && depth >= d) ? Value::deeper() :
      (kind == AT_LEAST && v > k)  ? Value::atLeast(v) :
      (kind == AT_MOST  && v <= k) ? Value::atMost(k)  :
      (kind == LOOP && v == k)     ? Value::loop(256)  :
      Value::nil();
  }

  Value toTT(int k, int d) PURE {
    return
      isDeeper() ? Value{DEEPER, k, d} :
      isLoop()   ? Value{LOOP,   k, 0} :
      isNil()    ? Value{AT_LEAST, -64, 0} :
      *this;
  }

private:
  union PackedValue {    
    u16 bits;
    struct {
      unsigned kind:2;
      unsigned value:7;
      unsigned depth:7;
    };
    
    PackedValue(const Value &v): kind(v.kind), value(v.v + 64), depth(v.depth) {}
    PackedValue(u16 bits): bits(bits) {}

    Value unpack() { return Value((Kind)kind, value - 64, depth); }
  };

public:
  u32 pack() PURE { return PackedValue(*this).bits; }
  static Value unpack(u16 bits) { return PackedValue(bits).unpack(); }
};

class Transtable {
#define SLOTS_BITS 34
  // 128GB : u64[1ull << SLOTS_BITS]
  u64 *slots; 

  class Data {
  public:
    Value up, down;
    
    Data(u32 data) : up(Value::unpack(data >> 16)), down(Value::unpack((u16) data)) {}
    
    Value getValue(int k, int depth) {
      Value v = up.fromTT(k, depth);
      return v.isNil() ? down.fromTT(k, depth) : v;
    }

    static u64 pack(Value up, Value down) { return (up.pack() << 16) | down.pack(); }
  };
  
public:
  Transtable() {
    slots = (u64 *) calloc(1<<(SLOTS_BITS - 17), 1<<20);
    assert(slots);
  }

  ~Transtable() {
    free(slots);
    slots = 0;
  }
  
  Value get(u64 h, int k, int depthPos, int maxDepth) {
    assert(depthPos <= maxDepth);
    u64 slot = slots[h >> (64 - SLOTS_BITS)];
    if ((u32) slot != (u32) h) { return Value::nil(); }
    return Data(slot >> 32).getValue(k, maxDepth - depthPos);
  }

  void put(u64 h, int k, int depthPos, int maxDepth, Value value) {
    assert(depthPos < maxDepth);
    assert(!value.isNil());
    if (value.isLoop() && value.depth < depthPos) { return; }
    u64 slot = slots[h >> (64 - SLOTS_BITS)];
    bool isMatch = (u32) slot == (u32) h;
    Data data(slot >> 32);
    Value ttv = value.toTT(k, maxDepth - depthPos);
    Value up   = (k >= 0) ? ttv : (isMatch ? data.up   : Value::nil());
    Value down = (k < 0)  ? ttv : (isMatch ? data.down : Value::nil());
    slots[h >> (64 - SLOTS_BITS)] = (Data::pack(up, down) << 32) | (u32) h;
  }
};

class History {
  struct HistHash { size_t operator()(u128 key) const { return (size_t) key; } };
  
  std::unordered_map<u128, int, HistHash> map;
  int level;
  
public:
  History(): level(0) {} 
  
  int pos(u128 key) {
    auto it = map.find(key);
    return (it == map.end()) ? -1 : it->second;
  }

  void push(u128 key) { map[key] = level++; }

  void pop(u128 key) {
    --level;
    assert(map[key] == level);
    map.erase(key);
  }
};

template<typename T>
class Bits {
  struct it {
    T bits;
    int operator*() { return firstOf(bits); }
    void operator++() { bits &= bits - 1; }
    bool operator!=(it o) { return bits != o.bits; }
  };

  T bits;
public:
  Bits(T bits) : bits(bits) {}
  it begin() { return {bits}; }
  it end()   { return {0}; }
};

template<typename T> inline Bits<T> bits(T v) { return Bits<T>(v); }

struct Eval {
  u64 pointsBlack;
  u64 pointsWhite;

  Value value(int k) {
    int n;
    return
      (pointsBlack && (n = 2 * size(pointsBlack) - TOTAL_POINTS) > k) ? Value::atLeast(n) :
      (pointsWhite && (n = TOTAL_POINTS - 2 * size(pointsWhite)) <= k) ? Value::atMost(n) :
      Value::nil();
  }
};

u64 bensonAlive(u64 black, u64 white);
bool canPlay(int pos, u64 black, u64 white);
u64 stonesBase3(u64 stones);

class Node {
  u64 black, white;
  int koPos;
  int _nPass;
  bool swapped;
  
public:
  Node(): black(0), white(0), koPos(-1), _nPass(0), swapped(false) {}
  Node(u64 black, u64 white, int koPos) : black(black), white(white), koPos(koPos), _nPass(0), swapped(false) {}
  Node(const char *board) : Node() {
    int y = 0;
    int x = 0;
    for (;*board; ++board) {
      char c = *board;
      if (c == '|' || c == '\n') {
        ++y; x = 0;
      } else {
        int p = P(y, x);
        if (c == 'x') {
          SET(p, black);
        } else if (c == 'o') {
          SET(p, white);
        } else if (c == 'k') {
          koPos = p;
        } else {
          assert(c == '.');
        }
        ++x;
      }
    }
  }
  
  operator string() {
    // printf("%lx %lx\n", black, white);
    char buf[128];
    char *out = buf;
    for (int y = 0; y < SIZE; ++y) {
      for (int x = 0; x < SIZE; ++x) {
        int p = P(y, x);
        assert(!(IS(p, black) && IS(p, white)));
        *out++ = IS(p, black) ? 'x' : IS(p, white) ? 'o' : p == koPos ? 'k' : '.';
      }
      *out++ = '\n';
    }
    *out++ = 0;
    return buf;
  }

  bool operator==(const Node &n) {
    return black == n.black && white == n.white && koPos == n.koPos && _nPass == n._nPass &&
      swapped == n.swapped;
  }
  
  bool isKo() PURE { return koPos >= 0; }
  int nPass() PURE { return _nPass; }
  
  u64 positionBits() { return (stonesBase3(black) << 1) + stonesBase3(white); }
  
  u64 situationBits() {
    assert(koPos >= -1 && _nPass >= 0);
    return (koPos + 1) | (_nPass << 6) | (swapped ? 0x100 : 0);
  }
  
  Eval eval() PURE {
    return Eval{bensonAlive(black, white), bensonAlive(white, black)};
  }
  
  u64 genMoves(const Eval &eval) PURE;
  
  Node play(int p) PURE {
    Node n(*this);
    n.playAux(p);
    n.swap();
    n.rotate();
    return n;
  }

private:
  bool canPlay(int pos) PURE {
    assert(pos >= 0);
    return (pos == koPos) ? false : (pos == PASS) ? (nPass() <= 1) :
      ::canPlay(pos, black, white);
  }
  
  void setGroup(int pos, int gid);
  void playAux(int pos);
  void playNotPass(int pos);
  void swap() { std::swap(black, white); swapped != swapped; }
  void rotate();
};

#define F(y, x) IS(t(P(y, x)), stone)
int quadrantAux(u64 stone, auto t) {
  return
    (SIZE == 3) ? (F(0, 0) << 2) | (F(0, 1) + F(1, 0)) :
    (SIZE == 4) ? (F(1, 1) << 3) | ((F(0, 1) + F(1, 0)) << 1) | F(0, 0) :
    (SIZE == 5) ? (F(1, 1) << 5) | ((F(0, 2) + F(2, 0)) << 3) | (F(0, 0) << 2) | (F(0, 1) + F(1, 0)) :
    (F(2, 2) << 6) | ((F(0, 2) + F(2, 0)) << 4) | (F(1, 1) << 3) | ((F(0, 1) + F(1, 0)) << 1) | F(0, 0);
}

int diagAux(u64 stone, auto t) {
  return (SIZE <= 4) ? F(0, 1) : ((F(1, 2) << 2) | (F(0, 2) << 1) | F(0, 1));
}
#undef F

int quadrant(u64 black, u64 white, auto t) {
  return (quadrantAux(black | white, t) << 8) | quadrantAux(black, t);
}

int diagValue(u64 black, u64 white, auto t) {
  return (diagAux(black | white, t) << 8) | diagAux(black, t);
}

union Bytes {
  u64 value;
  u8 bytes[8];
  unsigned operator[](int p) { return bytes[p]; }
};

u64 reflectY(u64 stones) {
  Bytes b{.value = stones};
  for (int y = 0; y < SIZE / 2; ++y) { std::swap(b.bytes[y], b.bytes[(SIZE - 1) - y]); }
  return b.value;
}

#define LINE(i) (_pext_u64(stones, 0x0101010101010101ull << i) << (i * 8))
u64 transpose(u64 stones) {
  return LINE(0) | LINE(1) | LINE(2) | LINE(3) | LINE(4) | LINE(5);
}
#undef LINE

u64 reflectX(u64 stones) { return transpose(reflectY(transpose(stones))); }

void Node::rotate() {
#define R(x) (SIZE - 1 - x)
  auto ident  = [](int p) { return p; };
  auto reflX  = [](int p) { return P(Y(p), R(X(p))); };
  auto reflY  = [](int p) { return P(R(Y(p)), X(p)); };
  auto reflXY = [](int p) { return P(R(Y(p)), R(X(p))); };
  auto diag   = [](int p) { return P(X(p), Y(p)); };
#undef R
  
  //  |AB|
  //  |CD|
  int A = quadrant(black, white, ident);
  int B = quadrant(black, white, reflX);
  int C = quadrant(black, white, reflY);
  int D = quadrant(black, white, reflXY);

  if (max(C, D) > max(A, B)) {
    black = reflectY(black);
    white = reflectY(white);
    koPos = reflY(koPos);
    std::swap(A, C);
    std::swap(B, D);
  }
  if (max(B, D) > max(A, C)) {
    black = reflectX(black);
    white = reflectX(white);
    koPos = reflX(koPos);
    std::swap(A, B);
    std::swap(C, D);
  }
  assert(A >= max(max(B, C), D));
  if (diagValue(black, white, diag) > diagValue(black, white, ident)) {
    black = transpose(black);
    white = transpose(white);
    koPos = diag(koPos);    
  }
}

u64 extract(u64 stones) {
  Bytes b{.value = stones};
  return b[0] | (b[1] << 6) | (b[2] << 12) | (b[3] << 18) | (b[4] << 24) |
    (((u64) b[5]) << 30);
}

int tab3[256] = {
   0,    1,    3,    4,    9,   10,   12,   13,   27,   28,   30,   31,   36,   37,   39,   40,
  81,   82,   84,   85,   90,   91,   93,   94,  108,  109,  111,  112,  117,  118,  120,  121,
 243,  244,  246,  247,  252,  253,  255,  256,  270,  271,  273,  274,  279,  280,  282,  283,
 324,  325,  327,  328,  333,  334,  336,  337,  351,  352,  354,  355,  360,  361,  363,  364,
 729,  730,  732,  733,  738,  739,  741,  742,  756,  757,  759,  760,  765,  766,  768,  769,
 810,  811,  813,  814,  819,  820,  822,  823,  837,  838,  840,  841,  846,  847,  849,  850,
 972,  973,  975,  976,  981,  982,  984,  985,  999, 1000, 1002, 1003, 1008, 1009, 1011, 1012,
1053, 1054, 1056, 1057, 1062, 1063, 1065, 1066, 1080, 1081, 1083, 1084, 1089, 1090, 1092, 1093,
2187, 2188, 2190, 2191, 2196, 2197, 2199, 2200, 2214, 2215, 2217, 2218, 2223, 2224, 2226, 2227,
2268, 2269, 2271, 2272, 2277, 2278, 2280, 2281, 2295, 2296, 2298, 2299, 2304, 2305, 2307, 2308,
2430, 2431, 2433, 2434, 2439, 2440, 2442, 2443, 2457, 2458, 2460, 2461, 2466, 2467, 2469, 2470,
2511, 2512, 2514, 2515, 2520, 2521, 2523, 2524, 2538, 2539, 2541, 2542, 2547, 2548, 2550, 2551,
2916, 2917, 2919, 2920, 2925, 2926, 2928, 2929, 2943, 2944, 2946, 2947, 2952, 2953, 2955, 2956,
2997, 2998, 3000, 3001, 3006, 3007, 3009, 3010, 3024, 3025, 3027, 3028, 3033, 3034, 3036, 3037,
3159, 3160, 3162, 3163, 3168, 3169, 3171, 3172, 3186, 3187, 3189, 3190, 3195, 3196, 3198, 3199,
3240, 3241, 3243, 3244, 3249, 3250, 3252, 3253, 3267, 3268, 3270, 3271, 3276, 3277, 3279, 3280,
};

#define POW8 6561
#define POW16 43046721ull
#define POW24 (POW16 * POW8)
#define POW32 (POW16 * POW16)
u64 base3(u64 bits) {
  Bytes b{.value = bits};
  return tab3[b[0]] + POW8*tab3[b[1]] + POW16*tab3[b[2]] + POW24*tab3[b[3]] + POW32*tab3[b[4]];
}

u64 stonesBase3(u64 stones) { return base3(extract(stones)); }

u64 groupAt(int pos, u64 black) {
  assert(pos >= 0 && IS(pos, black));
  const u64 saveBlack = black;
  CLEAR(pos, black);
  u64 open = 0;
  while (true) {
    for (int p : NEIB(pos)) { if (p >= 0 && IS(p, black)) { CLEAR(p, black); SET(p, open); } }
    if (!open) { break; }
    pos = POP(open);
  }
  return saveBlack ^ black;
}

// returns captured *black* group at pos.
u64 capture(int pos, u64 black, u64 white) {
  assert(pos >= 0 && IS(pos, black));
  u64 empty = INSIDE & ~(black | white);
  const u64 saveBlack = black;
  CLEAR(pos, black);
  u64 open = 0;
  while (true) {
    for (int p : NEIB(pos)) { if (p >= 0) {
        if (IS(p, empty)) { return 0; }
        if (IS(p, black)) { CLEAR(p, black); SET(p, open); }
      }
    }
    if (!open) { break; }
    pos = POP(open);
  }
  return saveBlack ^ black;
}

bool canPlay(int pos, u64 black, u64 white) {
  assert(pos >= 0 && IS(pos, INSIDE));
  if (IS(pos, black | white)) { return false; }
  SET(pos, black);
  for (int p : NEIB(pos)) {
    if (p >= 0 && IS(p, white) && capture(p, white, black)) { return true; }
  }
  return !capture(pos, black, white);
}

void Node::playNotPass(int pos) {
  // assert(pos >= 0 && pos != koPos && IS(pos, INSIDE & ~(black|white)));
  assert(::canPlay(pos, black, white));
  _nPass = 0;
  bool maybeKo = true;
  u64 captured = 0;
  SET(pos, black);
  for (int p : NEIB(pos)) {
    if (p >= 0) {
      if (IS(p, white)) {
        captured |= capture(p, white, black);
      } else if (IS(p, INSIDE)) {
        maybeKo = false;
      }
    }
  }
  koPos = (maybeKo && size(captured) == 1) ? firstOf(captured) : -1;
  white &= ~captured;
}

void Node::playAux(int pos) {
  assert(!(isKo() && nPass()));  // Can't have Ko after pass.
  if (pos == PASS) {
    if (isKo()) {
      koPos = -1;
    } else {
      assert(nPass() <= 1);
      ++_nPass;
    }
  } else {
    playNotPass(pos);
  }
}

u64 Node::genMoves(const Eval &eval) const {
  assert(nPass() < 2);
  u64 moves = (INSIDE & ~(black | white) & ~(eval.pointsBlack | eval.pointsWhite)) | PASS;
  for (int p : bits(moves)) { if (!canPlay(p)) { CLEAR(p, moves); } }
  return moves;
}

template<typename T, int N>
class vect {
  T v[N];
  int _size = 0;

public:
  void push(T t) { assert(_size < N); v[_size++] = t; }
  T pop()        { assert(_size > 0); return v[--_size]; }
  int size() { return _size; }
  bool isEmpty() { return _size <= 0; }
  bool has(T t) {
    for (T e : *this) { if (e == t) { return true; } }
    return false;
  }
  void clear() { _size = 0; }
  
  T *begin() { return v; }
  T *end() { return v + _size; }
  T operator[](int i) { return v[i]; }
};

struct Region {
  u64 area;
  u32 vital;
  u32 border;

  // Region(u64 area, u32 vital, u32 border) : area(area), vital(vital), border(border) {}
  
  bool isCoveredBy(u32 gidBits) { return (border & gidBits) == border; }
  // int size() { return ::size(area); }
  bool isUnconditional(u32 aliveGids) {
    return isCoveredBy(aliveGids) && (vital || size(area) < 8);
  }
};

u32 aliveGroups(vect<Region, 10> &regions) {
  while (true) {
    u32 alive = 0;
    u32 halfAlive = 0;
    for (Region &r : regions) {      
      if (u32 vital = r.vital) {
        for (int g : bits(vital)) {
          if (IS(g, halfAlive)) {
            SET(g, alive);
          } else {
            SET(g, halfAlive);
          }
        }
      }
    }
    if (!alive) { return 0; }
    bool changed = false;
    for (Region &r : regions) {
      if (r.vital && !r.isCoveredBy(alive)) {
        r.vital = 0;
        changed = true;
      }
    }
    if (!changed) { return alive; }
  }
}

// returns black unconditionally alive groups and unconditionally controlled points.
u64 bensonAlive(u64 black, u64 white) {
  vect<Region, 10> regions;
  u64 empty = INSIDE & ~(black | white);
  u64 emptyNotSeen = empty;
  u8 gids[48] = {0};
  // int nextGid = 1;
  while (emptyNotSeen) {
    int pos = firstOf(emptyNotSeen);
    // u64 area = 1ull << pos;
    u64 inside = INSIDE;
    CLEAR(pos, inside);
    u64 open = 0;
    u32 vital = -1;
    u32 border = 0;
    while (true) {
      u32 neibGroups = 0;
      for (int p : NEIB(pos)) {
        if (p >= 0) {
          if (IS(p, black)) {
            int gid = gids[p];
            if (!gid) {
              gid = p + 1;
              for (int pp : bits(groupAt(p, black))) { gids[pp] = gid; }
            }
            SET(gid, neibGroups);
          } else if (IS(p, inside)) { CLEAR(p, inside); SET(p, open); }
        }
      }
      border |= neibGroups;
      if (IS(pos, empty)) { vital &= neibGroups; }
      if (!open) { break; }
      pos = POP(open);
    }
    emptyNotSeen &= inside;
    regions.push(Region{INSIDE ^ inside, vital, border});
  }
  u64 points = 0;
  if (u32 aliveGids = aliveGroups(regions)) {
    for (Region &r : regions) { if (r.isUnconditional(aliveGids)) { points |= r.area; } }
    for (int gid : bits(aliveGids)) { points |= groupAt(gid - 1, black); }                        
    // for (int p : bits(black)) { if (IS(gids[p], aliveGids)) { SET(p, points); } }
  }
  return points;
}

Transtable tt;

Value max(History history, Node node, int k, int depthPos, int maxDepth) {
  u64 position  = node.positionBits();
  u64 sbits = node.situationBits();
  u128 situation = (((u128) sbits) << 64) | position;
  bool isPlain = !node.isKo() && !node.nPass();
  
  if (int historyPos = history.pos(situation)) {
    assert(historyPos < depthPos);
    return Value::loop(historyPos);
  }
  
  Value v = isPlain ? tt.get(position, k, depthPos, maxDepth) : Value::nil();
  if (v.isFinalEnough(k)) { return v; }

  Eval eval = node.eval();
  v = eval.value(k);
  if (v.isFinalEnough(k) || depthPos >= maxDepth) { return v; }

  history.push(situation);
  u64 moves = node.genMoves(eval);
  v = Value::nil();
  for (int move : bits(moves)) {
    Node sub = node.play(move);
    Value subValue = max(history, sub, -k-1, depthPos + 1, maxDepth).negate();
    v.accumulate(subValue, k);
    if (v.isCut(k)) { break; }
  }
  history.pop(situation);
  assert(v.isFinalEnough(k));
  if (isPlain) { tt.put(position, k, depthPos, maxDepth, v); }
  return v;
}

#include <stdio.h>

void testBasics() {
  assert(firstOf((u32)1) == 0);
  assert(firstOf((u64)2) == 1);
  assert(size((u32)0) == 0);
  assert(size((u64)-1) == 64);
  assert(IS(0, 1));
  assert(!IS(1, 1));
  u64 bits = 10;
  assert(POP(bits) == 1);
  assert(bits == 8);
  CLEAR(3, bits);
  assert(bits == 0);
  SET(2, bits);
  assert(bits == 4);

  assert(P(0, 1) == 1);
  assert(P(1, 0) == 8);
  assert(X(10) == 2);
  assert(Y(10) == 1);
}

void testValue() {
  Value v = Value::nil();
  assert(v.isNil() && !v.isLoop() && !v.isAtLeast(0) && !v.isAtMost(0) && !v.isDeeper());
  v.accumulate(Value::atMost(1), 3);
  assert(v.isAtMost(1) && v.isAtMost(2));
  v.accumulate(Value::atMost(2), 3);
  // printf("%s\n", S(v));
  assert(!v.isAtMost(1) && v.isAtMost(2));
  
  v = Value::atLeast(1);
  assert(v.isAtLeast(0) && v.isAtLeast(1) && !v.isAtMost(1) && !v.isAtMost(0) && !v.isDeeper());

  v = Value::unpack(Value::deeper().toTT(2, 3).pack());
  assert(v.isDeeper() && v.v == 2 && v.depth == 3);

  // printf("v1 %s; v2 %s\n", S(v), S(v2));
  Transtable tt;
  u64 h = 1;
  assert(tt.get(h,   0, 10, 11).isNil());
  assert(tt.get(h, -64,  0, 20).isNil());

  tt.put(h,  0, 1, 20, Value::deeper());
  assert(tt.get(h, 0, 5, 20).isDeeper());
  assert(tt.get(h, 1, 5, 20).isNil());
  assert(tt.get(h, 0, 0, 20).isNil());
  
  tt.put(h, -2, 1, 20, Value::atLeast(-1));
  assert(tt.get(h, 0, 5, 20).isDeeper());
  assert(tt.get(h, 1, 5, 20).isNil());
  assert(tt.get(h, 0, 0, 20).isNil());
  
  assert(tt.get(h, -2, 3, 20).isAtLeast(-1));
  assert(tt.get(h, -1, 3, 20).isNil());
  assert(tt.get(h, -3, 3, 20).isAtLeast(-2));
}

u64 fromString(const char *s) {
  u64 stones = 0;
  int y = 0;
  int x = 0;
  for (const char *ptr = s; *ptr; ++ptr) {
    if (*ptr == '|') {
      ++y;
      x = 0;
    } else {
      if (*ptr == 'x' || *ptr == 'o') {
        SET(P(y, x), stones);
      } else {
        assert(*ptr == '.');
      }
      ++x;
    }
  }
  return stones;
}

void testBenson() {
  assert(fromString("") == 0);
  assert(fromString("xx") == 3);
  assert(fromString("...|x") == 0x100);
  
  if (SIZE == 3) {
    u64 black = fromString(".x.|xx.|...");
    assert(bensonAlive(black, 0) == 0);
    u64 white = fromString("...|...|..o");
    assert(bensonAlive(black, white) == 0x070707);
    assert(bensonAlive(fromString(".x.|.x.|.x."), 0) == 0x070707);
    assert(bensonAlive(0, 0) == 0);
  }
}

void testCanPlay() {
  assert(canPlay(0, 0, 0));
  assert(!canPlay(0, 0, fromString(".o.|o..")));
  assert(canPlay(0, fromString("...|xo."), fromString(".o.|...")));
  assert(canPlay(0, fromString("...|xx."), fromString(".ox|ox.")));
}

void testRotate() {
  //Node n(fromString(".x.|, fromString("..o"), P(1, 2));
  //n.rotate();
  Node n(".xo|.o.|..o");
  // printf("%s\n", STR(n));
  n = n.play(P(1, 2));
  // printf("%s\n", STR(n));
  assert(n == Node("xok|.xo"));
}

int main() {
  testBasics();
  testValue();
  testBenson();
  testCanPlay();
  testRotate();
}









/*
// set gids for the black group at pos.
void Node::setGroup(int pos, int gid) {
  assert(pos >= 0 && IS(pos, black));
  u64 seen = 1ull << pos;
  u64 open = 0;
  while (true) {
    gids[pos] = gid;
    for (int p : NEIB(pos)) {
      if (p >= 0 && IS(p, black) && !IS(p, seen)) { SET(p, seen); SET(p, open); }
    }
    if (!open) { break; }
    pos = POP(open);
  }
}
*/
