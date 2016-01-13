/*
union TTValue {
  u16 bits;
  struct {
    int v:8;
    bool inside:1;
    unsigned a:7;
  };

  Value getValue(
};
*/

class Value {  
  Value(Kind kind, int v, int limit): kind(kind), v(v), limit(limit) {}
  Value(Kind kind, int v): Value(kind, v, 256);
  
public:
  int v;

  static Value nil()    { return Value(NIL, 0); }
  static Value deeper() { return Value(DEEPER, 0); }
  // static Value atMost(int k)  { return Value(AT_MOST, k); }
  // static Value atLeast(int k) { return Value(AT_LEAST, k); }

  static const char *kindName(Kind kind) {
    switch (kind) {
      // case OUTSIDE: return "outside";
      // case INSIDE: return "inside";
      //case AT_LEAST: return "at least";
      //case AT_MOST: return "at most";
    case NORMAL: return "";
    case DEEPER: return "deeper";
      // case LOOP: return "loop";
    case NIL: return "nil";
    default: return "?";
    }
  }
  
  operator string() const {
    char buf[64];
    snprintf(buf, sizeof(buf), "V(%s, %d, %d)", kindName(kind), v, limit);
    return buf;
  }

  // bool operator==(const Value &o) { return kind == o.kind && v == o.v && limit == o.limit; }
  
  bool isAtLeast(int k) PURE {
    assert(k != 0);
    return (k > 0) ? (kind == OUTSIDE && v >= k) : ((kind == OUTSIDE && v > 0) || (kind == INSIDE && v <= 0 && v >= k-1));
  }

  bool isAtMost(int k)  PURE {
    assert(k != 0);
    return (k > 0) ? ((kind == OUTSIDE && v < 0) || (kind == INSIDE && v >= 0 && v <= k+1)) : (kind == OUTSIDE && v <= k);
  }

  bool isDeeper()       PURE { return kind == DEEPER; }
  bool isNil()          PURE { return kind == NIL; }
  
  bool isFinalEnough(int k) PURE { return isDeeper() || isAtLeast(k + 1) || isAtMost(k); }
  bool isCut(int k) PURE { return isAtLeast(k + 1); }

  Value negate() PURE { return Value(kind, -v, limit); }

  void accumulate(const Value &sub, int k) {
    assert(!isCut(k));

    
    
    if (sub.isCut(k) || isNil()) {
      *this = sub;
    } else {
      limit = min(limit, sub.limit);
      if (sub.isDeeper()) {
        *this = sub;
      } else if(!isDeeper()) {
        
      }
    }
    
    if (sub.isCut(k) || sub.isDeeper() || isNil()) {
      *this = sub;
    } else if (!isDeeper()) {
      assert(sub.kind == AT_MOST && kind == AT_MOST);
      v = max(v, sub.v);  // atMost(max(v, sub.v));
    }
  }

  void max(const Value &old) {
    if (old.isNil() || old.isDeeper() || isDeeper()) { return; }
    assert(kind == AT_LEAST || kind == AT_MOST);
    assert(old.kind == AT_LEAST || old.kind == AT_MOST);
    if (kind == AT_LEAST && old.kind == AT_LEAST) {
      v = max(v, old.v);
    } else if (kind == AT_MOST && old.kind == AT_MOST) {
      v = min(v, old.v);
    } else {
      assert(kind != old.kind);
      if (kind == AT_LEAST) {
        assert(v <= old.v && v < 0);
      } else {
        assert(old.v <= v && v > 0);
      }
      v = 0;  // marker: between -k and k.
    }
  }

  static Value and(Value a, Value b) {
    
  }

  Value fromTT(int k, int d) PURE {
    return
      (kind == DEEPER && v == k && depth >= d) ? Value::deeper() :
      (kind == AT_LEAST && v > k)  ? Value::atLeast(v) :
      (kind == AT_MOST  && v <= k) ? Value::atMost(k)  :
      //      (kind == LOOP && v == k)     ? Value::loop(256)  :
      Value::nil();
  }

  Value toTT(int k, int d) PURE {
    assert(!isNil());
    return
      isDeeper() ? Value{DEEPER, k, d} :
    // isLoop()   ? Value{LOOP,   k, 0} :
      *this;
      // isNil()    ? Value{AT_LEAST, -64, 0} :
  }

private:
  union PackedValue {    
    u16 bits;
    struct {
      unsigned kind:2;
      unsigned value:7;
      unsigned depth:7;
    };
    
    PackedValue(const Value &v): kind(v.kind), value(v.v + 64), depth(v.depth) {
      assert(v.kind >= 0 && v.kind < 4 && v.depth < 128);
    }
    PackedValue(u16 bits): bits(bits) {}

    Value unpack() { return Value((Kind)kind, value - 64, depth); }
  };

public:
  u32 pack() PURE { return PackedValue(*this).bits; }
  static Value unpack(u16 bits) { return PackedValue(bits).unpack(); }
};

union Packed {
  u16 bits;
  struct {
    unsigned kind:2;
    unsigned a:7;
    unsigned b:7;
  };
};

u16 pack(unsigned kind, unsigned a, unsigned b) {
  Packed p{.kind=kind, .a=a, .b=b};
  return p.bits;
}












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
u64 bensonAlive(u64 black, u64 white, int *gids, u64 *shadows) {
  vect<Region, 10> regions;
  u64 empty = INSIDE & ~(black | white);
  u64 emptyNotSeen = empty;
  // u8 gids[48] = {0};
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
            SET(gids[p], neibGroups);
          } else if (IS(p, inside)) {
            CLEAR(p, inside);
            SET(p, open);
          }
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
    for (int gid : bits(aliveGids)) { points |= black & shadows[gid]; }
    // groupAt(gid - 1, black); }                        
    // for (int p : bits(black)) { if (IS(gids[p], aliveGids)) { SET(p, points); } }
  }
  return points;
}

/*
unsigned gidsNeib(int pos, u64 color) {
  assert(pos >= 0);
  u64 g = 0;
  for (int p : NEIB(pos)) {
    if (p >= 0 && IS(p, color)) {
      SET(gids[p], g);
    }
  }
  return g;
}
*/

/*
int readGroups(u64 black, u64 *shadows) {
  u64 *out = shadows;
  while (black) {
    int pos = firstOf(black);
    u64 open = 0;
    u64 shadow = 1ull << pos;
    while (true) {
      for (int p : NEIB(pos)) {
        if (p >= 0 && !IS(p, shadow)) {
          SET(p, shadow);
          if (IS(p, black)) { SET(p, open); }
        }
      }
      if (!open) { break; }
      pos = POP(open);
    }
    black &= ~shadow;
    // assert(out < end);
    *out++ = shadow;
  }
  return (int)(out - shadows);
}

int readGroups(u64 black, int base, int *gids, u64 *shadows) {
  int n = readGroups(black, shadows + base);
  // assert(shadows + base + n < shadowsEnd);
  for (int i = base; i < base + n; ++i) {
    for (int p : bits(black & shadows[i])) { gids[p] = i; }
  }
  return n;
}
*/


  /*
    int ypos = Y(pos);
    int xpos = X(pos);
    int centerPlay = min(min(ypos, SIZE - 1 - ypos), min(xpos, SIZE - 1 - xpos));
  */


/*
int readGroups(const u64 black, const u64 white, u64 *shadows, u64 *end) {
  int nBlack = readAux(black, white, shadows, end);
  int nWhite = readAux(white, black, shadown + nBlack, end);
  return readAux(readAux(0, black, white, gids, libs), white, black, gids, libs);
}
*/

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
