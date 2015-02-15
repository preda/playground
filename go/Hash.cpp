// (c) Copyright 2014 Mihai Preda. All rights reserved.

#include "Hash.hpp"
#include "transtable.hpp"
#include "data.hpp"

#include <stdio.h>

#define C(a, b) ((uint128_t)a << 64) + b

const uint128_t zob0[64] = {
C(0x82eb5db7e76eda91, 0x0d0436070f0dd158), C(0x501db094b41a5c2b, 0xfcac18bcf0eaedf4), 
C(0x70b0b64c0745b77d, 0x85b161ea05a5e791), C(0x479a0e3a38aa3ad0, 0x4bf950e81543de96), 
C(0xe523d6941bc5895a, 0xd2283235ebaa7d8e), C(0x956f8e0097fcd381, 0xcce4fe2b4f66e211), 
C(0x2991a3629fce411b, 0xd14998bc6f0bc55c), C(0x985750d704d57faf, 0xa3ddc3168c61ac64), 
C(0xb3a8b5a98ad68573, 0x42d0dea53a8e810a), C(0xce1a9ddfee918aee, 0x8fa33f07e10dc186), 
C(0xc8ae81455da41761, 0xf8b37d6b70f4c98d), C(0x949d271e414a84c9, 0x6905f7611084f23a), 
C(0x302229edb2ada5e3, 0xe41755f26430bb91), C(0xc58dc95148d02fee, 0x162658191703240e), 
C(0xdacbf8df917bf3ff, 0x0447f26ce5141496), C(0xf7e5c6ac0f3da9fb, 0xf565194c72926cb3), 
C(0x41acdc11ddd0bacc, 0x66dc2304899ed46a), C(0xad120721aa20dd32, 0xdfeba9ce31251d2d), 
C(0x5f756766e69666c2, 0xeef9b21b607f1dec), C(0xe45889eaaff9818e, 0x184fa9c5b926b5eb), 
C(0xeb15a141b5a0fe1a, 0x27c27d8c80668178), C(0x62fae48f68fac6ac, 0x5ed45e0b43044047), 
C(0x8e2aa2a20ab7f65b, 0xe05f730c1598066a), C(0x0ab0ede7f54ea8bb, 0xd0d454636ae6c0a0), 
C(0xacf05e08e95bba78, 0x96187a2155ab8e29), C(0x5010b4813946ce66, 0x8d18a8fe422b21dc), 
C(0x46d3978a13c46867, 0x728c08e85d5e3ad0), C(0xa539374c5701437e, 0x4a9b190e3dd2f858), 
C(0xac754fb762e3bfe4, 0xb3f41c5f56ea19dc), C(0x99a4610c19371d23, 0x555600a422a26357), 
C(0x52d82621fce232e9, 0xbc5d45821409a404), C(0x39847b1d00d0c9a8, 0x95127251663cd140), 
C(0x4b2b57ee651e43c2, 0xe0b75eca2d4cd3d8), C(0x45512e1f9260338a, 0x3a06e58df3ea122e), 
C(0x4a1b4970f1c9889a, 0x36be3322d7307c2e), C(0xc09e069920e73257, 0xf60d34d8a0580ac4), 
C(0x118294387b26877f, 0x6bea143588f6800e), C(0x219853d1db06bc78, 0xa50366bb92533967), 
C(0x4b7cf2760f42e210, 0x9d8161df9ca5260b), C(0xb409e19c4ea8f935, 0xa8b86df447fbf9e1), 
C(0x76c91e383d109cd8, 0x6911a0c3eceb6c54), C(0x175a1ad0189f535c, 0xf812518d79c1da96), 
C(0x064ad1f960403c35, 0xcaf5f00b7b77e76b), C(0xb91665addd4b0868, 0x1e91c1875c634a2b), 
C(0xd0f48cf78f4b36d2, 0x748cd419eb939653), C(0x26fd7b63712d2b6a, 0xb54444e54d9d6566), 
C(0xbe6dbfc1f3ceeaa9, 0x50d55be1d4d0986f), C(0x4f5faf4677847bfe, 0x6d6ca2991f3509cb), 
C(0xab58f76994b1ee08, 0x5a64c2eef185f891), C(0x3f92499eccfa656a, 0xe5532977e602f3ce), 
C(0x3de48656d57af123, 0x5bf43d6411bdd5a1), C(0xcd8d928c31e8efce, 0xf57a5e2ee5768f89), 
C(0x5b01e370155dfc44, 0x8f105e17cd4dfb43), C(0xf18983d5a804cd45, 0x69b2ca67a9110d1a), 
C(0x3e52f02d27bd1d9f, 0x2e16782f3f67a586), C(0x3061cf043c5dd16f, 0x78acc56f91638f1a), 
C(0xbae54d5516f37abf, 0xce3c6d9c624a2a8f), C(0x55bd0d733bc486ca, 0xadafe6f598fb7fb8), 
C(0xd12b2ee69c977600, 0xe79d6efd6cb0389f), C(0xc36c269689e0aa8f, 0x18adc00730c32c1d), 
C(0xf3ac65b1a4eddf57, 0x93bbaac3241c1e92), C(0xde04f6fd371e1643, 0x3983bf951e130338), 
C(0x9286593ac48df076, 0xc608bb9fdd1c56f9), C(0x58a59d1973c4cf2e, 0xa72a4a0c985df036),
};

const uint128_t zob1[64] = {
C(0x36150ebc38e29ac6, 0x44b2d4a666f90401), C(0xebab6b1ac1b00e41, 0x51ba2ed5e6c81c22), 
C(0xf34b52f8a6970bf7, 0xddad3ae3b1c936a9), C(0xa9edcb1c858e7ff2, 0x829a4ce8c105d28a), 
C(0x4ce34227a3fcb57f, 0x627dffea30310918), C(0x7e99478735851857, 0x75c359a82160d86a), 
C(0x0fea41629343bd3e, 0x7d0941a8db7e0fc9), C(0x030c701b30a339b0, 0x3fee735eeb6aeb7e), 
C(0xbcedb992016deae6, 0x224ce768ec9b40aa), C(0xe84e5e742a8cd0a3, 0x8a9089dba30325ef), 
C(0xb3b3eecfd414c009, 0x8aed27b6a172cc42), C(0xd2828e9dd20fd2f1, 0x986b61bc9b94312e), 
C(0xef1dd542245e73b8, 0xf8ed850615d9df29), C(0x95dedfd40801ec40, 0xf9a961bb4109ce21), 
C(0x613dc63a69aa5904, 0x3e67550142212e67), C(0x6164cfcba7a4e3ef, 0xd233987a8e1e5e83), 
C(0xfc6acfc31e647906, 0x4fc1eabdf0261d01), C(0x42f30218648c7111, 0x2ce5d9cd53760a0a), 
C(0x10947619ebd86ff8, 0x506e04bb7ea0f099), C(0x34fc05fcc36ef351, 0xff11067714026a56), 
C(0x5fa54f0723dceed5, 0x53a9dce8c4480a7b), C(0x89b857601efd78a4, 0xed30a5c8930b2bda), 
C(0xafebaac033505b43, 0x6b1d7d8dd49c5274), C(0x8bffaec8b57e07e3, 0xe4d25abfe0fcea3a), 
C(0x427fab0db9b8b5e2, 0x6bedfa03d812d6e6), C(0x6d23214488e815ff, 0xa5e8dee520db48ee), 
C(0xaf1d347aa00ee76a, 0xeb0e2c696feb11e6), C(0x5e427b4876b46ff7, 0xb651c3f26b918c83), 
C(0x88b87b909c198c20, 0x2a6eda56f18ffe7e), C(0x9ad132b949c49516, 0x7668703b48bb2e74), 
C(0x78fdf019a3aaf813, 0x2241472599d987ed), C(0x2479d0818c784bf2, 0x77cb3e12ed1af064), 
C(0x29b6830184eabf46, 0x24a0b13526cfbfd8), C(0xef8258824c315155, 0x17aa40ea4ab975e6), 
C(0xe3920ce1843a774f, 0x35b892f1ccb6a8de), C(0x2f4df1394259bb3d, 0xb6a3e0e8f03471d7), 
C(0x58555d5ccdf36632, 0xc3b8232258769f8f), C(0x0cc9a6ff9f0647c5, 0x33a5d422ff49d329), 
C(0xc05ea5c0dd1de96d, 0x59afb96827ba25d4), C(0xf889afebdc6505eb, 0xd1a484008c540c3e), 
C(0x0a2a3ad9b4e5d076, 0x6a6d559e87d6abbb), C(0x5f2318d67f79463b, 0x4d18be366038a7c5), 
C(0x10e927932d6bda53, 0x9875f24b44bb4a8c), C(0x0bff46d6c1a0e680, 0x420ff2c29ab67637), 
C(0x8f29b9596f6f9f49, 0x74ad58b877af0fb1), C(0xa7ece212cbb23ed7, 0x85ec42ddde86b02a), 
C(0x0c2aaeb500c9b15e, 0x6a989b558da3111b), C(0x89b1b7725f4d70d0, 0x8def0777a365777c), 
C(0x7acdbac6954f1d82, 0xd2a4b255e920c5b7), C(0x745c049e2e78f5aa, 0x135f535f05634ae9), 
C(0x1741961060e1b9d0, 0xcbcaad1c42969f3d), C(0xef74e2c151be7894, 0x9c8b7fd5f4ef607d), 
C(0xd84963b7e1f38e8f, 0x3d2dafa9f812d4cb), C(0xae97e1fa8a2dc082, 0xbe211b3ca26e1343), 
C(0x0e5dc65dc6e6f2e2, 0xa6d0f5b24e5d9b94), C(0x088e52a05bd9a1d9, 0x2916729946fff18b), 
C(0x703597e8485e1d49, 0x6434b646190a2c45), C(0xf42e840fba31dd1a, 0xb00e150dd8d62099), 
C(0x9d8de5c186567497, 0xfe896e218b063ad2), C(0x7b5bb2af7f5ab1cb, 0xdf0ac30daa6ce7ec), 
C(0xbb270c4d6d935cd2, 0x58bb1c16fbde8944), C(0x2d8c16116ecc9904, 0xfbdc1d312b8e9e4b), 
C(0xe81ad6be178691cb, 0xf1e285dafebc8bd1), C(0xc3a001e636ab3474, 0x844008b2ad2ba5c5), 
};

#undef C

template<bool BLACK> uint128_t hashPos(int p) { return BLACK ? zob0[p] : zob1[p]; }

uint128_t hashSide()          { return hashPos<true>(0); }
uint128_t hashPass(int nPass) { return hashPos<true>(nPass); }

Hash::Hash(uint128_t hash, uint128_t situationHash) :
  hash(hash),
  situationHash(situationHash),
  pos(hash & MASK),
  lock(((uint64_t) (hash >> 64)) & LOCK_MASK) {
  while (pos >= SIZE) { pos >>= RES_BITS; }
}

template<bool BLACK> Hash Hash::update(int pos, int oldNPass, int nPass, uint64_t capture) const {
  uint128_t situationDelta = hashSide();
  if (pos != PASS) { situationDelta ^= hashPos<BLACK>(pos); }
  if (capture) {
    for (int p : Bits(capture)) {
      situationDelta ^= hashPos<!BLACK>(p);
    }
  }  
  uint128_t hashDelta = situationDelta;
  if (oldNPass)    { hashDelta ^= hashPass(oldNPass); }
  if (nPass)       { hashDelta ^= hashPass(nPass); }
  return Hash(hash ^ hashDelta, situationHash ^ situationDelta);
}

void Hash::print() {
  printf("Hash %lx %lx ", pos, lock);
}

template Hash Hash::update<true>(int, int, int, uint64_t) const;
template Hash Hash::update<false>(int, int, int, uint64_t) const;
