int readZeropad(global int *in, int N, int line, int p) { return (p & (1 << (N - 1))) ? 0 : readC(in, N - 1, line, p); }

void sq2(double *v) {
  double a = v[0];
  v[0] = (a + v[1]) * (a - v[1]);
  v[1] *= 2 * a;
}

void sq4(double *v) {
  double b[] = {v[1], v[3]};
  v[1] = v[2];
  v[2] = v[0] - b[0];
  v[3] = v[1] - b[1];
  sq2(v);
  sq2(v + 2);
  sq2(b);
  v[2] = v[0] + b[0] - v[2];
  v[3] = v[1] + b[1] - v[3];
  v[0] -= b[1];
  v[1] += b[0];
}

void sq8(double *v) {
  double b[4];
  for (int i = 0; i < 4; ++i) { b[i] = v[2 * i + 1]; } // v.odd
  for (int i = 1; i < 4; ++i) { v[i] = v[2 * i]; }     // v.even
  for (int i = 0; i < 4; ++i) { v[i + 4] = v[i] - b[i]; }
  sq4(v);
  sq4(v + 4);
  sq4(b);
  for (int i = 0; i < 4; ++i) { v[i + 4] = v[i] + b[i] - v[i + 4]; }
  v[0] -= b[3];
  v[1] += b[0];
  v[2] += b[1];
  v[3] += b[2];
}

void swapLDS(local double *lds, double *save, uint gs, uint me) {
  // Swap LDS and 'save'.
  for (int i = 0; i < 8; ++i) {
    double x = lds[i * gs + me];
    lds[i * gs + me] = save[i];
    save[i] = x;
  }
}

// Helper for sq64(). "save" is unchanged on return.
void half64(uint me, local double *lds, double *save) {
  // FFT(8), DIF 3 rounds, wrong (transposed) LDS access pattern.
  for (int round = 2; round >= 0; --round) {
    for (int i = 0; i < 4; ++i) {
      FFT_SETUP_CORE(3, round, 1, i);
      double a = read(lds, 3, me, line);
      double b = read(lds, 3, me, line + mr);
      write( a + b, lds, 3, me, line);
      writeT(a - b, lds, 3, me + e, line + mr);
    }
  }

  swapLDS(lds, save, 8, me);
  sq8(save);
  swapLDS(lds, save, 8, me);

  for (int round = 0; round < 3; ++round) {
    for (int i = 0; i < 4; ++i) {
      FFT_SETUP_CORE(3, round, 1, i);
      double a = read( lds, 3, me, line);
      double b = readT(lds, 3, me + e, line + mr);
      write(a + b, lds, 3, me, line);
      write(a - b, lds, 3, me, line + mr);
    }
  }
}

// 8 threads: 'me' in [0, 7].
// LDS is 8x8; FFT lines are 'vertical' (the 'wrong' layout).
void sq64(uint me, local double *lds) {
  double save[8];
  // Save LDS to 8 VREGs/lane.
  for (int i = 0; i < 8; ++i) {
    int p = i - me;
    double x = lds[(p & 7) * 8 + me];
    save[i] = p >= 0 ? x : -x;
  }

  half64(me, lds, save);
  swapLDS(lds, save, 8, me);
  half64(me, lds, save);

  {
    double x = save[7];
    for (int i = 7; i >= 0; --i) { save[i] += save[i - 1]; }
    save[0] -= x;
  }
  
  for (int i = 0; i < 8; ++i) {
    double x = readT(lds, 3, i + me, me);
    save[i]           += x;
    save[(i + 1) & 7] += (i < 7) ? -x : x;
  }

  for (int i = 0; i < 8; ++i) {
    write(save[i], lds, 3, i, me);
  }
}

void half2k(local double *lds, uint me) {
  for (int round = 4; round >= 0; --round) {
    for (int i = 0; i < 4; ++i) {
      FFT_SETUP_CORE(6, round, 1, me / 64 + i * 4);
      uint p = me & 63;
      double a = read(lds, 6, line, p);
      double b = read(lds, 6, line + mr, p);
      write( a + b, lds, 6, line, p);
      writeC(a - b, lds, 6, line + mr, p + e);
    }
  }

  sq64(me & 7, lds + me / 8 * 64);

  for (int round = 0; round < 5; ++round) {
    for (int i = 0; i < 4; ++i) {
      FFT_SETUP_CORE(6, round, 1, me / 64 + i * 4);
      uint p = me & 63;
      double a = read( lds, 6, line, p);
      double b = readC(lds, 6, line + mr, p + e);
      write(a + b, lds, 6, line, p);
      write(a - b, lds, 6, line + mr, p);
    }
  }
}

KERNEL(256) void sq2k(global double *tab) {
  local double lds[2048];
  uint me = get_local_id(0);
  global double *base = tab + get_group_id(0) * 2048;
  
  for (int i = 0; i < 8; ++i) {
    write(base[cut8(i * 256 + me)], lds, 6, me & 31, me / 32 + i * 8);
  }

  double save[8];
  for (int i = 0; i < 8; ++i) {
    int line = i * 4 + me / 64;
    int p    = me & 63 - 2 * line;
    double x = read(lds, 6, line, p & 63);
    save[i] = (p >= 0) ? x : -x;
  }
  
  half2k(lds, me);
  
  swapLDS(lds, save, 256, me);
  
  half2k(lds, me);
  
  for (int i = 0; i < 8; ++i) {
    int line = i * 4 + me / 64;
    int p = me & 63;
    double x = readC(lds, 6, line, p + 2 * line + 1);
    writeC(save[i] - x, lds, 6, line, p + 1);
    x = read(lds, 6, line, p);
    base[cut8(i * 256 + me)] = save[i] + x;
  }
}

