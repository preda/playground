#include <assert.h>
#include <stdint.h>
#include <immintrin.h>
#include <stdio.h>

#include <unordered_map>
#include <string>
#include <utility>
#include <algorithm>

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
  SIZE = 4,
  N = SIZE * SIZE,
  MAX_GROUPS = 18,
  DELTA = 8,
  MOVE_PASS = 63,
  PASS_0 = 56,
  PASS_1 = 57,
  PASS_2 = 58,
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

class Transtable {
#define HASH_BITS 58
#define SLOTS_BITS 35
#define BLOCK_BITS 32
#define LOCK_BITS 26
#define LOCK_MASK ((1 << LOCK_BITS) - 1)
#define NOT_FOUND (-128)
  // 128GB == 2^37 == u32[1 << 35]
  u32 *slots;
  int counter = 0;
public:
  Transtable() {
    slots = (u32 *) calloc(1 << 25, 1 << 12);
    assert(slots);
  }

  ~Transtable() {
    free(slots);
    slots = 0;
  }
  
  int get(u64 hash) {
    /* hash has 58 bits. Lower 26 bits form the 'lock', and the higher 32 form the block index.
       A slot has 4bytes. 8 consecutive slots form a block.
       A slot matches if its lower 26 bits match the lock.
    */
    
    assert((hash >> HASH_BITS) == 0);  // no more than HASH_BITS
    u32 lock = ((u32) hash) & LOCK_MASK;
    for (u32 *p = slots + ((hash >> 26) << 3), *end = p + 8; p < end && *p; ++p) {
      if ((*p & LOCK_MASK) == lock) { return (*p >> LOCK_BITS) - 32; }
    }
    return NOT_FOUND;
  }

  void put(u64 hash, int value) {
    assert((hash >> HASH_BITS) == 0);
    assert(value >= -31 && value <= 31);
    u32 lock = ((u32) hash) & LOCK_MASK;
    u32 newSlot = ((value + 32) << LOCK_BITS) | lock;
    u32 *block = slots + ((hash >> 26) << 3);
    for (u32 *p = block, *end = p + 8; p < end; ++p) {
      if ((*p == 0) || ((*p & LOCK_MASK) == lock)) {
        int oldValue = (*p >> LOCK_BITS) - 32;
        assert(value != oldValue);
        // printf("%d %d\n", oldValue, value);
        assert(oldValue == -32 || value == 0 || (oldValue > 0 && value > oldValue) || (oldValue < 0 && value < oldValue));
        *p = newSlot;
        return;
      }
    }
    block[counter] = newSlot;
    counter = (counter + 1) & 7;
  }
};

class History {
  // struct HistHash { size_t operator()(u128 key) const { return (size_t) key; } };
  
  std::unordered_map<u64, int> map;
  int level;
  
public:
  History(): level(0) {} 
  
  int pos(u64 key) {
    auto it = map.find(key);
    return (it == map.end()) ? -1 : it->second;
  }

  void push(u64 key, int depth) { assert(depth == level); map[key] = level++; }

  void pop(u64 key, int depth) {
    --level;
    assert(map[key] == level && depth == level);
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

u64 stonesBase3(u64 stones);
int scoreEmpty(u64 empty, u64 black, u64 white);

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

class Node {
  u64 black, white;
  u64 blackAlive, whiteAlive;
  int koPos;
  
public:
  Node(): black(0), white(0), blackAlive(0), whiteAlive(0), koPos(PASS_0) {}
  
  Node(u64 black, u64 white, int koPos) :
    black(black), white(white), blackAlive(0), whiteAlive(0),
    koPos(koPos) {}

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

  char symbolAtPos(int pos) const {
    assert(pos >= 0);
    if (pos == koPos) {
      assert(!IS(pos, black | white | blackAlive | whiteAlive));
      return 'k';
    } else if (IS(pos, blackAlive)) {
      return IS(pos, black) ? 'X' : IS(pos, white) ? 'c' : '+';
    } else if (IS(pos, whiteAlive)) {
      return IS(pos, white) ? 'O' : IS(pos, black) ? 'y' : '0';
    } else {
      return IS(pos, black) ? 'x' : IS(pos, white) ? 'o' : '.';
    }
  }
  
  operator string() const {
    char buf[128];
    char *out = buf;
    for (int y = 0; y < SIZE; ++y) {
      for (int x = 0; x < SIZE; ++x) {
        int p = P(y, x);
        assert(!(IS(p, black) && IS(p, white)));
        *out++ = symbolAtPos(p);
      }
      *out++ = '\n';
    }
    *out++ = 0;
    return buf;
  }

  bool operator==(const Node &n) {
    return black == n.black && white == n.white && koPos == n.koPos;
  }
  
  u64 getBlack() PURE { return black; }
  u64 getWhite() PURE { return white; }
  int getKoPos() PURE { return koPos; }
  
  u64 position() PURE {
    u64 smallPosition = (stonesBase3(black) << 1) + stonesBase3(white);
    assert(!(smallPosition >> 58));
    assert(koPos >= 0 && koPos <= PASS_2);
    return (((u64) koPos) << 58) | smallPosition;
  }
    
  Node play(int p) PURE {
    assert(p >= 0 && p != koPos);
    Node n(*this);
    n.playAux(p);
    n.swap();
    n.rotate();
    return n;
  }

  bool updateAlive();
  int value(int k) PURE;
  vect<u8, N+1> genMoves() PURE;
  void rotate();
  
private:
  void setGroup(int pos, int gid);
  void playAux(int pos);
  void playNotPass(int pos);
  void swap() {
    std::swap(black, white);
    std::swap(blackAlive, whiteAlive);
  }
  int valueOfMove(int pos) PURE;
};

struct Region {
  u64 area;
  u32 vital;
  u32 border;

  bool isCoveredBy(u32 gidBits) { return (border & gidBits) == border; }
  bool isUnconditional(u32 aliveGids) {
    return isCoveredBy(aliveGids) && (vital || size(area) < 8);
  }
};

extern inline Region regionAt(int pos, u64 black, u64 blackAlive, u64 empty, u8 *gids) {
  assert(pos >= 0 && IS(pos, empty) && !IS(pos, blackAlive));
  u64 iniActive = INSIDE & ~blackAlive;
  u64 active = iniActive;
  CLEAR(pos, active);
  u64 open = 0;
  u32 vital = -1;
  u32 border = 0;
  while (true) {
    u32 neibGroups = 0;
    for (int p : NEIB(pos)) {
      if (p >= 0) {
        if (IS(p, black)) {
          SET(gids[p], neibGroups);
        } else if (IS(p, active)) {
          CLEAR(p, active);
          SET(p, open);
        }
      }
    }
    border |= neibGroups;
    if (IS(pos, empty)) { vital  &= neibGroups; }
    if (!open) { break; }
    pos = POP(open);
  }
  return Region{iniActive ^ active, vital, border};
}

u32 aliveGroups(vect<Region, 10> &regions) {
  while (true) {
    u32 alive = 0;
    u32 halfAlive = 0;
    for (Region &r : regions) {
      if (u32 vital = r.vital) {
        alive |= halfAlive & vital;
        halfAlive |= vital;
      }
    }
    bool changed = false;
    if (alive) {
      for (Region &r : regions) {
        if (r.vital && !r.isCoveredBy(alive)) {
          r.vital = 0;
          changed = true;
        }
      }
    }
    if (!changed) { return alive; }
  }
}

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

void readGids(u64 black, u8 *gids) {
  while (black) {
    int pos = POP(black);
    int gid = pos;
    gids[pos] = gid;    
    u64 open = 0;
    while (true) {
      for (int p : NEIB(pos)) {
        if (p >= 0 && IS(p, black)) {
          CLEAR(p, black);
          SET(p, open);
          gids[p] = gid;
        }
      }
      if (!open) { break; }
      pos = POP(open);
    }
  }
}

// update whiteAlive
bool Node::updateAlive() {
  u8 whiteGids[48];
  readGids(white, whiteGids);
  u64 emptyNotSeen = INSIDE & ~(black | white | whiteAlive);
  vect<Region, 10> regions;
  while (emptyNotSeen) {
    Region r = regionAt(firstOf(emptyNotSeen), white, whiteAlive, emptyNotSeen, whiteGids);
    emptyNotSeen &= ~r.area;
    regions.push(r);
  }
  if (u32 aliveGids = aliveGroups(regions)) {
    for (Region &r : regions) { if (r.isUnconditional(aliveGids)) { whiteAlive |= r.area; } }
    for (int gid : bits(aliveGids)) { whiteAlive |= groupAt(gid, white); }
    return true;
  }
  return false;
}

int Node::value(int k) const {
  k = abs(k);
  if (koPos == PASS_2) {
    u64 emptyUnsettled = INSIDE & ~(black | white | blackAlive | whiteAlive);
    int scoreUnsettled = scoreEmpty(emptyUnsettled, black, white);
    int score = scoreUnsettled + size(black | blackAlive) - size(white | whiteAlive);
    return (score >= k || score <= -k) ? score : 0;
  } else {
    int atLeast = 2 * size(blackAlive) - N;
    if (atLeast >= k) { return atLeast; }    
    int atMost = N - 2 * size(whiteAlive);
    if (atMost <= -k) { return atMost; }
    if (atMost < k && atLeast > -k) { return 0; }
    return -128;
  }
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
  // printf("x %lx %lx\n", saveBlack, black);
  return saveBlack ^ black;
}

// whether the *black* group at pos is in atari.
bool isAtari(int pos, u64 black, u64 white) {
  assert(pos >= 0 && IS(pos, black));
  u64 empty = INSIDE & ~(black | white);
  CLEAR(pos, black);
  u64 open = 0;
  int nLibs = 0;
  while (true) {
    for (int p : NEIB(pos)) {
      if (p >= 0) {
        if (IS(p, empty)) {
          ++nLibs;
          if (nLibs >= 2) { return false; }
          CLEAR(p, empty);
        }
        if (IS(p, black)) {
          CLEAR(p, black);
          SET(p, open);
        }
      }
    }
    if (!open) { break; }
    pos = POP(open);
  }
  assert(nLibs == 1);
  return true;
}

int Node::valueOfMove(int pos) const {
  u64 empty = INSIDE & ~(black | white);
  int value = 0;
  bool isSuicide = true;
  int nBlack = 0;
  int nWhite = 0;
  int nEmpty = 0;
  for (int p : NEIB(pos)) {
    if (p >= 0) {
      if (IS(p, empty)) {
        ++nEmpty;
        isSuicide = false;
        // printf("empty %d ", p); 
      } else if (IS(p, INSIDE)) {
        if (IS(p, black)) {
          ++nBlack;
          if (isAtari(p, black, white)) {
            value += 2;
          } else {
            isSuicide = false;
            // printf("not atari %d ", p);
          }
        } else {
          assert(IS(p, white));
          ++nWhite;
          if (isAtari(p, white, black)) {
            value += 3;
            isSuicide = false;
            // printf("white atari %d ", p);
          }
        }
      }
    }
  }
  if (nBlack + nWhite + nEmpty == 4) {
    if (nBlack == 1) {
      value += 1;
    } else if (nBlack == 2) {
      if ((IS(pos-1, black) && IS(pos+1, black)) ||
          (IS(pos-DELTA, black) && IS(pos+DELTA, black))) {
        value += 2;
      }
    }
    
    if (nWhite == 1) {
      value += 1;
    } else if (nWhite == 2) {
      if ((IS(pos-1, white) && IS(pos+1, white)) ||
          (IS(pos-DELTA, white) && IS(pos+DELTA, white))) {
        value += 2;
      }
    }
    
    value += nEmpty;
  } else {
    value += nEmpty + nBlack;
  }
  return isSuicide ? 0 : (value + 1);
}
  
vect<u8, N+1> Node::genMoves() const {
  assert(koPos < PASS_2);
  int tmp[N];
  int n = 0;    
  u64 moves = (INSIDE & ~(black | white) & ~(blackAlive | whiteAlive));
  if (koPos < PASS_0) { CLEAR(koPos, moves); }    
  for (int p : bits(moves)) {
    if (int v = valueOfMove(p)) { tmp[n++] = (v << 8) | p; }
    // if (!canPlay(p, black, white)) { CLEAR(p, moves); }
  }
  std::sort(tmp, tmp + n);
  vect<u8, N+1> ret;
  if (koPos == PASS_1) { ret.push(MOVE_PASS); }
  for (int *p = tmp + n - 1; p >= tmp; --p) { ret.push(*p & 0xff); }
  if (koPos != PASS_1) { ret.push(MOVE_PASS); }
  return ret;
}

#define F(y, x) IS(t(P(y, x)), stone)
int quadrantAux(u64 stone, auto t) {
  int a = 0;
  switch (SIZE) {
  case 6: a |= F(2, 2) << 8;
  case 5: a |= ((F(1, 2) + F(2, 1)) << 6) | ((F(0, 2) + F(2, 0)) << 4);
  case 4: a |= F(1, 1) << 3;
  case 3: a |= ((F(0, 1) + F(1, 0)) << 1) | F(0, 0);
  }
  return a;
}

int diagAux(u64 stone, auto t) {
  return (SIZE <= 4) ? F(0, 1) : ((F(1, 2) << 2) | (F(0, 2) << 1) | F(0, 1));
}
#undef F

int quadrant(u64 black, u64 white, auto t) {
  return (quadrantAux(black | white, t) << 9) | quadrantAux(black, t);
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

u64 reflectXT(u64 stones) { return reflectY(transpose(stones)); }

#define APPLY(f)                                        \
  black = f(black);                                     \
  white = f(white);                                     \
  blackAlive = blackAlive ? f(blackAlive) : blackAlive; \
  whiteAlive = whiteAlive ? f(whiteAlive) : whiteAlive

void Node::rotate() {
#define R(x) (SIZE - 1 - x)
  auto ident   = [](int p) { return p; };
  auto Ry      = [](int p) { return P(R(Y(p)), X(p)); };
  auto RxT     = [](int p) { return P(R(X(p)), Y(p)); };
  auto RxTInv  = [](int p) { return P(X(p), R(Y(p))); };
  auto RyxT    = [](int p) { return P(R(X(p)), R(Y(p))); };
  auto T       = [](int p) { return P(X(p), Y(p)); };
#undef R
  
  //  |AB|
  //  |CD|
  int A = quadrant(black, white, ident);
  int B = quadrant(black, white, RxTInv);
  int C = quadrant(black, white, Ry);
  int D = quadrant(black, white, RyxT);

  if (max(C, D) > max(A, B)) {
    APPLY(reflectY);
    koPos = (koPos < PASS_0) ? Ry(koPos) : koPos;
    std::swap(A, C);
    std::swap(B, D);
  }
  if (B > A) {
    APPLY(reflectXT);
    koPos = (koPos < PASS_0) ? RxT(koPos) : koPos;
    std::swap(A, B);
    std::swap(C, D);
  }
  assert(A >= max(max(B, C), D));
  if (diagValue(black, white, T) > diagValue(black, white, ident)) {
    APPLY(transpose);
    koPos = (koPos < PASS_0) ? T(koPos) : koPos;
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

int scoreEmpty(u64 empty, u64 black, u64 white) {
  assert(black | white);
  assert(!(black & white));
  int score = 0;
  while (empty) {
    int pos = POP(empty);
    int size = 1;
    u64 open = 0;
    bool touchesBlack = false, touchesWhite = false;
    while (true) {
      for (int p : NEIB(pos)) { if (p >= 0) {
          if (IS(p, empty)) {
            ++size;
            CLEAR(p, empty);
            SET(p, open);          
          } else if (IS(p, black)) {
            touchesBlack = true;
          } else if (IS(p, white)) {
            touchesWhite = true;
          }
        }
      }
      if (!open) { break; }
      pos = POP(open);
    }
    assert(touchesBlack || touchesWhite);
    if (!(touchesBlack && touchesWhite)) {
      score += touchesBlack ? size : -size;
    }   
  }
  return score;
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
  bool maybeKo = true;
  u64 captured = 0;
  SET(pos, black);
  for (int p : NEIB(pos)) {
    if (p >= 0) {
      if (IS(p, white)) {
        captured |= capture(p, white, black);
      } else if (IS(p, INSIDE)) {
        maybeKo = false;
        if (IS(p, blackAlive) && !IS(pos, blackAlive)) {
          assert(IS(p, black));
          blackAlive |= groupAt(pos, black);
        }
      }
    }
  }
  if (captured) {
    white &= ~captured;
    if (maybeKo && size(captured) == 1) {
      koPos = firstOf(captured);
    }
  }
  if (capture(pos, black, white)) {
    printf("pos %d ko %d\n%s\n", pos, koPos, STR(*this));
    CLEAR(pos, black);
    printf("vom %d\n", valueOfMove(pos));
    SET(pos, black);
  }
  assert(!capture(pos, black, white));  
}

void Node::playAux(int pos) {
  if (pos == MOVE_PASS) {
    if (koPos < PASS_0) {
      koPos = PASS_0;
    } else {
      assert(koPos < PASS_2);
      ++koPos;
    }
  } else {
    assert(pos >= 0 && pos != koPos);
    assert(IS(pos, INSIDE & ~(black|white)));
    koPos = PASS_0;
    playNotPass(pos);
  }
}

bool isFinalEnough(int v, int k) {
  return v != NOT_FOUND
    && ((k < 0 && (v <= k || v >= 0))
        || (k > 0 && (v >= k || v <= 0)));
}

bool isCut(int v, int k) {
  return (v > k) || (k > 0 && v >= k);
}

Transtable tt;

int maxMove(History &history, Node &node, int k, int depth, int *outCacheAt) {
  u64 bigPosition  = node.position();
  
  *outCacheAt = 1024;
  
  int historyPos = history.pos(bigPosition);
  if (historyPos >= 0) {
    assert(depth > historyPos);
    // if (!((depth - historyPos) & 1)) {
    *outCacheAt =  historyPos;
    // }    
    return 0;
  }

  printf("depth %d hash %lx\n%s\n", depth, bigPosition, STR(node));
  
  u64 smallPosition = bigPosition & ((1ull << 58) - 1);
  bool isPlain = (node.getKoPos() == PASS_0);
  if (isPlain) {    
    int v = tt.get(smallPosition);
    if (isFinalEnough(v, k)) { return v; }
  }

  node.updateAlive();
  // printf("Alv k %d, d %d\n%s value %s\n", k, depth, STR(node), STR(node.value(k)));
  
  int v = node.value(k);
  if (isFinalEnough(v, k)) {
    if (isPlain) { tt.put(smallPosition, v); }
    return v;
  }

  // if (node.getKoPos() >= PASS_2) { printf("v %d k %d\n%s\n", v, k, STR(node)); }
  assert(node.getKoPos() < PASS_2);
  
  history.push(bigPosition, depth);

  auto moves = node.genMoves();
  v = -128;
  int cacheAt = 1024;
  for (int move : moves) {
    if (depth == 0 && move == MOVE_PASS) { continue; } // avoid initial PASS.
    Node sub = node.play(move);
    int subCacheAt = 1024;
    int subValue = -maxMove(history, sub, -k, depth + 1, &subCacheAt);
    if (isCut(subValue, k)) {
      if (!isCut(v, k) || subCacheAt > cacheAt) {
        v = subValue;
        cacheAt = subCacheAt;
      }
      if (cacheAt == 1024) { break; }
    } else if (!isCut(v, k)) {
      v = max(v, subValue);
      cacheAt = min(cacheAt, subCacheAt);
    }
  }
  
  history.pop(bigPosition, depth);
  assert(isFinalEnough(v, k));
  if (isPlain && depth <= cacheAt) {
    printf("tt put depth %d v %d\n%s\n", depth, v, STR(node));
    tt.put(smallPosition, v);
  }
  // if (depth <= 2) { printf("Put k %d, d %d, %s\n%s\n", k, depth, STR(v), STR(node)); }
  *outCacheAt = cacheAt;
  return v;
}

void mtdf() {
  Node node;
  int k = 1;
  History history;
  while (true) {
    int cacheAt = 1024;
    int v = maxMove(history, node, k, 0, &cacheAt);
    printf("k %d, cacheAt %d, value %d\n", k, cacheAt, v);
    if (v >= k) {
      k = v + 1;
    } else {
      break;
    }
  }
}

bool testAll();

int main() {
  Node n(0x3f3f3f3f3f3full, 0, PASS_0);
  printf("%lx\n", n.position());
  
  assert(testAll());
  mtdf();
}



// --- test ---


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

/*
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
*/

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

  /*
  if (SIZE == 3) {
    u64 black = fromString(".x.|xx.|...");
    assert(bensonAlive(black, 0) == 0);
    u64 white = fromString("...|...|..o");
    assert(bensonAlive(black, white) == 0x070707);
    assert(bensonAlive(fromString(".x.|.x.|.x."), 0) == 0x070707);
    assert(bensonAlive(0, 0) == 0);
  }
  */
}

void testCanPlay() {
  assert(canPlay(0, 0, 0));
  assert(!canPlay(0, 0, fromString(".o.|o..")));
  assert(canPlay(0, fromString("...|xo."), fromString(".o.|...")));
  assert(canPlay(0, fromString("...|xx."), fromString(".ox|ox.")));
}

void testRotate() {
  if (SIZE == 3) {
    Node n(".xo|.o.|..o");
    // printf("%s\n", STR(n));
    n = n.play(P(1, 2));
    // printf("%s\n", STR(n));
    assert(n == Node("xok|.xo"));
  }

  if (SIZE == 4) {
    Node expected(".x..|oxxo|.o..|....");
    for (const char *s : {
        "....|..o.|oxxo|..x.",
        "..x.|oxxo|..o.|....",
      }){
      
      Node n(s);
      // printf("%s\n", STR(n));
      n.rotate();
      // printf("%s\n----\n\n", STR(n));
      assert(n == expected);
    }
  }
}

void testReadGroups() {
  if (SIZE == 4) {
    /*
    Node node("xxx.|ooxx|oox.|xooo");
    readGroups(node.getBlack(),
    */
    
    /*
    Node node("xxoo|.x..|xx.o|o.oo");
    int gids[64];
    int libs[16];
    int n = readGroups(node.getBlack(), node.getWhite(), gids, libs);
    assert(n == 4);
    for (int p : {P(0, 0), P(0, 1), P(1, 1), P(2, 0), P(2, 1)}) { assert(gids[p] == 1); }
    assert(libs[1] == 4);
    for (int p : {P(0, 2), P(0, 3)}) { assert(gids[p] == 2); }
    assert(libs[2] == 2);
    for (int p : {P(2, 3), P(3, 2), P(3, 3)}) { assert(gids[p] == 3); }
    assert(libs[3] == 3);    
    assert(gids[P(3, 0)] == 4);
    assert(libs[4] == 1);
    */
  }
}

bool testAll() {
  /*
  Node node("xxx.|ooxx|oox.|xooo");
  printf("%lx\n", capture(16, node.getWhite(), node.getBlack()));
  */

  /*
  Node node("xxox|xxo.|.xoo|o.oo");
  Eval eval(node);
  printf("eval %d\n", eval.valueOfMove(11));
  */

  testBasics();
  // testValue();
  testBenson();
  testCanPlay();
  testRotate();
  // testReadGroups();
  return true;
}
