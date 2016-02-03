/*
struct Test { unsigned p; u64 k; };

#include "tests.inc"

__global__ void test(unsigned p, u64 k) {
  U3 q = makeQ(p, k);
  testOut = expMod(p, q);
}

static void selfTest() {
  int n = sizeof(tests) / sizeof(tests[0]);
  for (Test *t = tests, *end = tests + n; t < end; ++t) {
    unsigned p = t->p;
    u64 k = t->k;
    int shift = __builtin_clz(p);
    assert(shift < 27);
    p <<= shift;
    // printf("p %u k %llu m: ", t->p, t->k); print(m);
    test<<<1, 1>>>(p, k);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(err));
      break;
    } else {
      if (testOut.a != 1 || testOut.b || testOut.c) {
        printf("ERROR %10u %20llu m ", t->p, t->k); print(m); print(out);
        break;
      } else {
        // printf("OK\n");
      }
    }
  }
}
*/

/*
bool launch(unsigned p, u64 k0, int t, unsigned *classes, int repeat) {
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    return true;
  }
  if (foundFactor) {
    printf("factor %llu\n", foundFactor);
    return true;
  }
  memcpy(classTab, classes, t * sizeof(unsigned));
  if (t < THREADS_PER_GRID) {
    printf("Tail %d\n", t);
    memset(classTab + t, 0xff, (THREADS_PER_GRID - t) * sizeof(unsigned));
  }
  tf<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(p, k0, k0);
  return false;
}

int findFactor(unsigned p, u64 k0, int repeat) {
  u64 timeStart = timeMillis();
  u64 time1 = timeStart;
  unsigned classes[THREADS_PER_GRID];
  int accepted = 0;
  int t = 0;
  int c = 0;
  int nLaunch = 0;
  for (; c < NCLASS; ++c) {
    if (acceptClass(p, c)) {
      classes[t++] = c;
      if (t >= THREADS_PER_GRID) {
        accepted += THREADS_PER_GRID;
        t = 0;
        ++nLaunch;
        if (launch(p, k0, THREADS_PER_GRID, classes, repeat)) { return -1; }
        if (!(nLaunch & 0xf)) {
          u64 time2 = timeMillis();
          printf("%8u: %u ms\n", c, (unsigned)(time2 - time1));
          time1 = time2;
        }
      }
    }
  }
  accepted += t;
  launch(p, k0, t, classes, repeat);
  u64 time2 = timeMillis(); time1 = time2;
  printf("%8u: %u ms; total %llu\n", c, (unsigned)(time2 - time1), time2 - timeStart);
  return accepted;
}
*/

    /*    
    u32 *pw = words + tid;
    int popCount = 0;
    for (u32 *p = words + tid, *end = words + NWORDS; p < end; p += THREADS_PER_BLOCK) {
      u32 w = ~*p;
      *p = w;
      popCount += __popc(w);
    }
    __syncthreads();
    u32 save = *words;
    u32 *countp = words;
    u32 *outPos = words + atomicAdd(countp, popCount);
    __syncthreads();
    *words = save;
    for (u32 *p = words + tid, *end = words + NWORDS; p < end; p += THREADS_PER_BLOCK) {
      u32 w = *p;
      extractBits(out, *p, ((p - words) << 5));
    }
    __syncthreads();
  }
  
      while (bits) {
        int bit = bfind(bits);
        bits &= ~(1<<bit);
        *outPos++ = bitBase + bit;
      }
      */  
  
  /*
  unsigned c = classTab[id];
  if (c == 0xffffffff) { return; }
  u64 k = k0 + c;
  for (int i = repeat; i > 0; --i) {
    if (isFactor(exp, k)) {
      printf("%d found factor %llu\n", id, k);
      deviceFactor = k;
      break;
    }
    k += NCLASS;
  }
  */


/*
__device__ int bumpBtc(int btc, u64 delta, u16 prime) {
  return ((btc -= delta % prime) < 0) ? (btc + prime) : btc; 
}

#define REPEAT_32(w, s) w(11)s w(13)s w(17)s w(19)s w(23)s w(29)s w(31)
#define REPEAT_64(w, s) w(37)s w(41)s w(43)s w(47)s w(53)s w(59)s w(61)
#define REPEAT(w, s) REPEAT_32(w, s)s REPEAT_64(w, s)
*/

/*
__device__ u16 invTab[ASIZE(primes)];
__global__ void initInvTab(u64 step) {
  int id = ID;
  u16 prime = primes[id];
  u16 inv = modInv16(step, prime);
  assert(inv == modInv32(step, prime));
  invTab[id] = inv;
}
*/

/*
__device__ u16 modInv16(u64 step, u16 prime) {
  u16 n = step % prime;
  u16 q = prime / n;
  u16 d = prime - q * n;
  int x = -q;
  int prevX = 1;
  while (d) {
    q = n / d;
    { u16 save = d; d = n - q * d; n = save;         }
    { int save = x; x = prevX - q * x; prevX = save; }
  }
  return (prevX >= 0) ? prevX : (prevX + prime);
}

__device__ u16 bitToClear(u32 exp, u64 k, u16 prime) {
  u64 step = 2 * NCLASS * (u64) exp;  
  u16 inv = modInv16(step, prime);
  assert(inv == modInv32(step, prime));
  return bitToClear(exp, k, prime, inv);
}

__device__ u16 bitToClear(u32 exp, u64 k, u16 prime, u16 inv) {
  u16 kmod = k % prime;
  u16 qmod = ((exp << 1) * (u64) kmod + 1) % prime;
  return (prime - qmod) * (u32) inv % prime;
}

__device__ u16 classBtc(u32 exp, u16 c, u16 prime, u16 inv) {
  u16 qInv = (c * (u64) (exp << 1) + 1) * inv % prime;
  u16 btc = qInv ? (prime - qInv) : qInv;
  assert(btc == bitToClear(exp, c, prime, inv));
  return btc;
}
*/


  /*
  p("u  ", u); p("xu ", (U3){x.c, x.d, x.e});
  U3 tmp = shr3wMul((U3) {x.c, x.d, x.e}, u);
  U3 d = mulLow(m, tmp);
  U3 r = (U3){x.a, x.b, x.c} - d;
  p("tmp ", tmp); p("x   ", (U3){x.a, x.b, x.c}); p("d   ", d); p("r   ", r);  
  return (U3){x.a, x.b, x.c} - mulLow(tmp, m);  
  */


    for (u64 *end = p + (NWORDS/2); p < end; ++p) {
      u64 w = ~*p;
      while (w) {
        u32 bit = currentWordPos + __builtin_ctzl(w);
        w &= (w - 1);
        // assert(bit < prev + 256);
        *out++ = (u8) (bit - prev);
        prev = bit;
      }
      currentWordPos += 64;
    }

/*
    int popc = 0;
    
    for (int i = tid; i < NWORDS; i += SIEV_THREADS) { popc += __popc(words[i] = ~words[i]); }
  
    u32 bits = words[tid];
    if (tid < 32) {
      words[0] = 0;
      words[1] = 0xffffffff;
    }
    __syncthreads();
    u32 pos = atomicAdd(words, popc | (1 << 20));
    atomicMin(words + 1, popc);
    __syncthreads();
    if (tid == 0) {
      words[0] = atomicAdd(pSize, words[0] & 0xfffff);
    }
    int min = words[1];
    pos = (pos & 0xfffff) - min * (pos >> 20);
    __syncthreads();
    int p = words[0] + tid;
    int i = tid;
    u32 delta = (tid + blockIdx.x * NWORDS) * 32;
    do {
      while (!bits) {
        bits = words[i += SIEV_THREADS];
        delta += SIEV_THREADS * 32;
      }
      int bit = bfind(bits);
      bits &= ~(1 << bit);
      kTab[p] = delta + bit;
      p += SIEV_THREADS;
    } while (--min);
    p += -tid + (int)pos;
    while (true) {
      while (!bits) {
        i += SIEV_THREADS;
        if (i >= NWORDS) { goto out; }
        bits = words[i];
        delta += SIEV_THREADS * 32;
      }
      int bit = bfind(bits);
      bits &= ~(1 << bit);
      kTab[p++] = delta + bit;
    }
  out:;
  } while (--rep);
}

__global__ void __launch_bounds__(SIEV_THREADS) sievA() { sieve(&kTabSizeA, kTabA); }
__global__ void __launch_bounds__(SIEV_THREADS) sievB() { sieve(&kTabSizeB, kTabB); }
*/

  /*
  printf("exp %u kStart %llu kEnd %llu k0Start %llu k0End %llu, %llu blocks %u, actual %u\n",
         exp, kStart, kEnd, k0Start, k0End, (k0Start + blocks * (u64) blockSize), blocks, 512 * 3 * 1024 / NWORDS);
  
  t1 = timeMillis();
  initInvTab<<<NPRIMES/1024, 1024>>>(exp);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERR;
  printf("initInvTab: %llu ms\n", timeMillis() - t1);
  t1 = timeMillis();
  kTabSizeA = 0;
  kTabSizeB = 0;

  int c = 992;
  u64 k = k0Start + c;
  foundFactor = 0;
  initBtcTab<<<NPRIMES/1024, 1024>>>(exp, k);
  sievA<<<SIEV_BLOCKS, SIEV_THREADS>>>();
  cudaDeviceSynchronize();
  CUDA_CHECK_ERR;
  int testBlocksA = testBlocks(kTabSizeA);
  kTabSizeA = 0;
  testA<<<testBlocksA, TEST_THREADS>>>(doubleExp, flushedExp, k);
  cudaDeviceSynchronize();
  if (foundFactor) { printf("Factor K: %llu\n", foundFactor); }
  CUDA_CHECK_ERR;
  return 0;
  */

  /*
  for (int cid = 0; cid < NGOODCLASS; ++cid) {
    int c = classTab[cid];
    u64 k = k0Start + c;

    int sizeB = kTabSizeB;
    int testBlocksB = testBlocks(sizeB);
    kTabSizeB = 0;
    initBtcTab<<<NPRIMES/1024, 1024>>>(exp, k);
    // if (!cid)
    sievA<<<SIEV_BLOCKS, SIEV_THREADS, 0, sieveStream>>>();
    usleep(100);
    if (testBlocksB) {
      testB<<<testBlocksB, TEST_THREADS, 0, testStream>>>(doubleExp, flushedExp, k);
    }
    //testB<<<1, 1, 0, testStream>>>(doubleExp, flushedExp, k);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR;
    int sizeA = kTabSizeA;
    int testBlocksA = testBlocks(sizeA);
    kTabSizeA = 0;
    // if (!cid)
    sievB<<<SIEV_BLOCKS, SIEV_THREADS, 0, sieveStream>>>();
    usleep(100);
    testA<<<testBlocksA, TEST_THREADS, 0, testStream>>>(doubleExp, flushedExp, k);
    // testA<<<1, 1, 0, testStream>>>(doubleExp, flushedExp, k);
    cudaDeviceSynchronize();
    u64 t2 = timeMillis();
    printf("%5d: class %5d: %llu; A %d (%d), B %d (%d)\n", cid, c, t2 - t1, sizeA, testBlocksA, sizeB, testBlocksB);
    t1 = t2;    
    if (foundFactor) { printf("Factor K: %llu\n", foundFactor); break; }
    CUDA_CHECK_ERR;
  }
  printf("Total time: %llu ms\n", timeMillis() - t0);
  cudaStreamDestroy(sieveStream);
  cudaStreamDestroy(testStream);
  CUDA_CHECK_ERR;
  */
}


/*

    initBtcTab<<<NPRIMES/1024, 1024>>>(exp, k);
    sieveAndTest<<<SIEV_BLOCKS, SIEV_THREADS>>>(doubleExp, flushedExp, k);
    cudaDeviceSynchronize();
    u64 t2 = timeMillis();
    printf("%llu\n", t2 - t1);
    t1 = t2;

  __global__ void __launch_bounds__(SIEV_THREADS, 4) sieveAndTest(u32 doubleExp, u32 flushedExp, u64 k) {
  __shared__ u32 words[NWORDS];
  const int tid = threadIdx.x;
  int rep = SIEV_REPEAT;
  while (true) {
    for (int i = 0; i < NWORDS / SIEV_THREADS; ++i) { words[tid + i * SIEV_THREADS] = 0; }
    __syncthreads();
    for (int i = tid; i < NPRIMES; i += SIEV_THREADS) {
      int prime = primes[i];
      int btc0  = btcTab[i];
      int btcAux = btc0 - (NCLASS * NBITS % prime) * blockIdx.x % prime;
      int btc = (btcAux < 0) ? btcAux + prime : btcAux;
      while (btc < NBITS) {
        atomicOr(words + (btc >> 5), 1 << (btc & 0x1f));
        btc += prime;
      }
    }
    __syncthreads();    
    u32 bits = ~words[tid];
    int i = tid;
    u32 delta = (tid + blockIdx.x * NWORDS) * 32;
    while (true) {
      while (!bits) {
        i += SIEV_THREADS;
        if (i >= NWORDS) { goto out; }
        bits = ~words[i];
        delta += SIEV_THREADS * 32;
      }
      int bit = bfind(bits);
      bits &= ~(1 << bit);
      U3 r = expMod(flushedExp, incMul(_U2(k + (delta + bit)), doubleExp));
      if (r.a == 1 && !(r.b | r.c)) { foundFactor = k; }
    }
  out:
    if (!--rep) { break; }
    __syncthreads();
  }
}
*/

/*
  int btc0  = btcTabs[0][i];
  int btcAux = btc0 - (NCLASS * NBITS % prime) * blockIdx.x % prime;
  int btc = (btcAux < 0) ? btcAux + prime : btcAux;
*/


--------------- below bit expand on cpu  --------

void extractBits(u64 *bits, u16 (*deltas)[TEST_ROWS * TEST_THREADS]) {
  u32 prev[TEST_THREADS];
  u32 *prevEnd = prev + TEST_THREADS;
  
  u64 *p = bits;
  for (int ci = 0; ci < NGOODCLASS; ++ci) {
    u16 *deltap = deltas[ci];
    u32 *prevp  = prev;

    memset(prev, 0, sizeof(prev));
    u32 currentWordPos = 0;

    for (u64 *end = p + (NWORDS/2); p < end; ++p) {
      u64 w = *p;
      while (w) {
        u32 bit = currentWordPos + __builtin_ctzl(w);
        w &= (w - 1);
        *deltap++ = (u16) (bit - *prevp);
        *prevp++ = bit;
        if (prevp == prevEnd) { prevp = prev; }
      }
      currentWordPos += 64;
    }
    assert(deltap + TEST_THREADS <= deltas[ci + 1]);
    memset(deltap, 0xff, sizeof(u16) * TEST_THREADS);
  }
}

  u64 *hostBits = 0;
  checkCuda(cudaHostAlloc(&hostBits, NGOODCLASS * NWORDS * 4, 0));
  time("Alloc hostBits");

  u16 (*deltas)[TEST_ROWS * TEST_THREADS];
  checkCuda(cudaHostAlloc(&deltas, NGOODCLASS * sizeof(deltas[0]), 0));
  time("Alloc deltas");
  cudaDeviceSynchronize(); time("init inv + btc");

    /*
    cudaMemcpyFromSymbol(hostBits, sievedBits, NGOODCLASS * NWORDS * 4, 0, cudaMemcpyDeviceToHost);
    CUDA_CHECK; time("Copy from device");
  
    extractBits(hostBits, deltas); time("Extract bits");

    cudaMemcpyToSymbol(kDeltas, deltas, NGOODCLASS * sizeof(deltas[0]));
    CUDA_CHECK; time("Copy to device");
    */

// Sieved bits are aggregated from shared memory after sieve() to this global memory block.
DEVICE u32 sievedBits[NGOODCLASS][NWORDS];

// Deltas of Ks for testing. This is a derivate of the sieved bits.
DEVICE u16 kDeltas[NGOODCLASS][TEST_ROWS * TEST_THREADS];




  /*
  int p1=-1, p2=-1;
  cudaDeviceGetStreamPriorityRange(&p1, &p2);
  CUDA_CHECK;
  printf("Priority %d %d\n", p1, p2);
  
  cudaStream_t sieveStream, testStream;
  cudaStreamCreateWithPriority(&sieveStream, cudaStreamNonBlocking, 0);
  CUDA_CHECK;
  cudaStreamCreateWithPriority(&testStream, cudaStreamNonBlocking, 1);
  CUDA_CHECK;
  */



// Among all the NCLASS classes, select the ones that are "good",
// i.e. not corresponding to a multiple of a small prime.
__global__ void initClasses(u32 exp) {
  __shared__ u32 pos;
  __shared__ u32 blockPos;
  pos = 0; __syncthreads();
  
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int myPos = -1;
  if (c < NCLASS && q1or7mod8(exp, c)
      && !multiple(exp, c, 3) && !multiple(exp, c, 5)
      && !multiple(exp, c, 7) && !multiple(exp, c, 11)) {
    myPos = atomicAdd(&pos, 1);
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
    blockPos = atomicAdd(&classTabPos, pos);
  }
  __syncthreads();

  if (myPos >= 0) {
    classTab[blockPos + myPos] = c;
  }
  /*
#ifndef NDEBUG
  __syncthreads();
  assert(pos == NGOODCLASS);
#endif
  */
}

