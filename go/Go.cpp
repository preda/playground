#include <assert.h>
#include <stdint.h>
#include <unordered_map>
#include <string>

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned u32;
typedef uint64_t u64;
typedef unsigned __int128 u128;

using std::string;

auto min(auto a, auto b) { return a < b  ? a : b; }
auto max(auto a, auto b) { return a >= b ? a : b; }

#define PURE const __attribute__((warn_unused_result))
#define S(x) ((string) x).c_str()

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

inline int firstOf(u64 bits) { return __builtin_ctzll(bits); }
inline int firstOf(u32 bits) { return __builtin_ctz(bits);   }

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

class Node {
  u64 black, white;
  int koPos;
  int nPass;
  bool swapped;
  byte gids[36];
  
public:
  Node(): black(0), white(0), koPos(0), nPass(0), swapped(false), gids(0) {}
    
  bool isKo() { return koPos; }
  int nPass() { return nPass; }
  
  u64 position() { return 0; }
  u64 situationBits() { return 0; }
  
  Value eval(int k) { return Value::nil(); }
  u64 genMoves() { return 0; }
  Node play(int p) { return *this; }
  
};

template<typename T> inline Bits<T> bits(T v) { return Bits<T>(v); }

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
