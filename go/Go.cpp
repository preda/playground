#include <assert.h>
#include <stdint.h>
#include <unordered_map>
#include <string>
#include <utility>

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
  SIZE_X = SIZE,
  SIZE_Y = SIZE,
  BIG_X = 8,
  BIG_Y = SIZE_Y + 1,
  DELTA = BIG_X,

  N = SIZE_X * SIZE_Y,
  BIG_N = BIG_X * BIG_Y,
  PASS = 63,
};

constexpr inline int P(int y, int x) { return (y << 3) + x; }
inline int Y(int pos) { return (pos >> 3); }
inline int X(int pos) { return pos & (BIG_X - 1); }

constexpr u64 insidePoints() {
  u64 ret = 0;
  for (int y = 0; y < SIZE_Y; ++y) {
    for (int x = 0; x < SIZE_X; ++x) {
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

inline bool isInside(int p) { return p >= 0 && IS(p, INSIDE); }

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

class Node {
  u64 black, white;
  int koPos;
  int _nPass;
  bool swapped;
  u8 gids[48];
  
public:
  Node(): black(0), white(0), koPos(0), _nPass(0), swapped(false), gids() {}
    
  bool isKo() { return koPos; }
  int nPass() { return _nPass; }
  
  u64 position() { return 0; }
  u64 situationBits() { return 0; }
  
  Value eval(int k) { return Value::nil(); }
  u64 genMoves() { return 0; }
  
  Node play(int p) PURE {
    Node n(*this);
    n.playAux(p);
    n.swap();
    return n;
  }

private:
  bool canPlay(int pos);
  void setGroup(int pos, int gid);
  void playAux(int pos);
  void playNotPass(int pos);
  void swap() { std::swap(black, white); swapped != swapped; }
  
  bool isWhite(int p) const { return p >= 0 && IS(p, white); }
  // bool isEmpty(int p) const { return isInside(p) && ~IS(p, black | white); }
  // u64 capture(int p) const;
};

// returns captured *black* group at pos.
u64 capture(int pos, u64 black, u64 white) {
  assert(pos >= 0 && IS(pos, black));
  u64 empty = INSIDE & ~(black | white);
  u64 seen = 1ull << pos;
  u64 open = 0;
  while (true) {
    for (int p : NEIB(pos)) {
      if (p >= 0) {
        if (IS(p, empty)) { return 0; }
        if (IS(p, black) && !IS(p, seen)) { SET(p, seen); SET(p, open); }
      }
    }
    if (!open) { return seen; }
    pos = POP(open);
  }
}

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

// Whether the *black* group at pos has any liberties.
// bool hasLiberty(int pos, u64 black, u64 white) { return !capture(pos, black, white); }

bool canPlay(int pos, u64 black, u64 white) {
  assert(pos >= 0 && IS(pos, INSIDE));
  if (IS(pos, black | white)) { return false; }
  SET(pos, black);
  for (int p : NEIB(pos)) {
    if (p >= 0 && IS(p, white) && capture(p, white, black)) { return true; }
  }
  return !capture(pos, black, white);
}

bool Node::canPlay(int pos) {
  assert(pos >= 0);
  if (pos == koPos) { return false; }
  if (pos == PASS) { return nPass() <= 1; }
  return ::canPlay(pos, black, white);
}

void Node::playNotPass(int pos) {
  // assert(pos >= 0 && pos != koPos && IS(pos, INSIDE & ~(black|white)));
  assert(::canPlay(pos, black, white));
  
  _nPass = 0;
  bool maybeKo = true;
  int newGid = -1;
  bool isSimple = true;
  u64 captured = 0;
  SET(pos, black);
  for (int p : NEIB(pos)) {
    if (p >= 0) {
      if (IS(p, black)) {
        maybeKo = false;
        if (newGid == -1) {
          newGid = gids[p];
        } else if (newGid != gids[p]) {
          isSimple = false;
        }
      } else if (IS(p, white)) {
        captured |= capture(p, white, black);
      } else if (IS(p, INSIDE)) {
        maybeKo = false;
      }
    }
  }
  koPos = (maybeKo && size(captured) == 1) ? firstOf(captured) : 0;
  white &= ~captured;
  if (isSimple) {
    if (newGid == -1) { newGid = pos; }
    gids[pos] = newGid;
  } else {
    assert(newGid >= 0);
    setGroup(pos, newGid);
  }
}

void Node::playAux(int pos) {
  assert(!(koPos && nPass()));  // Can't have Ko after pass.
  if (pos == PASS) {
    if (koPos) {
      koPos = 0;
    } else {
      assert(nPass() <= 1);
      ++_nPass;
    }
  } else {
    playNotPass(pos);
  }
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
u64 bensonAlive(u64 black, u64 white, u8 *gids) {
  vect<Region, 10> regions;
  u64 empty = INSIDE & ~(black | white);
  u64 emptyNotSeen = empty;
  while (emptyNotSeen) {
    int pos = firstOf(emptyNotSeen);
    u64 area = 1ull << pos;
    u64 open = 0;
    u32 vital = -1;
    u32 border = 0;
    while (true) {
      u32 neibGroups = 0;
      for (int p : NEIB(pos)) {
        if (p >= 0) {
          if (IS(p, black)) {
            SET(gids[p], neibGroups);
          } else if (IS(p, INSIDE) && !IS(p, area)) { // whiteOrEmpty && !area
            SET(p, area);
            SET(p, open);
          }
        }
      }
      border |= neibGroups;
      if (IS(pos, empty)) { vital &= neibGroups; }
      if (!open) { break; }
      pos = POP(open);
    }
    emptyNotSeen &= ~area;
    regions.push(Region{area, vital, border});
  }
  u64 points = 0;
  if (u32 aliveGids = aliveGroups(regions)) {
    for (int p : bits(black)) { if (IS(gids[p], aliveGids)) { SET(p, points); } }
    for (Region &r : regions) { if (r.isUnconditional(aliveGids)) { points |= r.area; } }
  }
  return points;
}

Transtable tt;

Value max(History history, Node node, int k, int depthPos, int maxDepth) {
  u64 position  = node.position();
  u64 sbits = node.situationBits();
  u128 situation = (((u128) sbits) << 64) | position;
  bool isPlain = !node.isKo() && !node.nPass();
  
  if (int historyPos = history.pos(situation)) {
    assert(historyPos < depthPos);
    return Value::loop(historyPos);
  }
  
  Value v = isPlain ? tt.get(position, k, depthPos, maxDepth) : Value::nil();
  if (v.isFinalEnough(k)) { return v; }
  
  v = node.eval(k);
  if (v.isFinalEnough(k) || depthPos >= maxDepth) { return v; }

  history.push(situation);
  u64 moves = node.genMoves();
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

void unitTest() {
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

int main() {
  unitTest();
}
