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
