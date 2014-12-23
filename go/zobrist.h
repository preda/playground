const uint64_t zob0[64] = {
0x54d918dd5d2f0090ull, 0xb75f37ff7f4ad6c2ull, 0x63d1c96eb2586f32ull, 0x03c5f87034eaeb43ull, 
0xe0bbd16d8c0920ceull, 0x8cfa0698245226eaull, 0x75d16a1fe38cc36dull, 0xd10f09a578f5a73cull, 
0x45cb961a9a71a545ull, 0x23c7d169df814c15ull, 0x84c52c5548c7ae71ull, 0x9ced066f2fb08692ull, 
0x604a9747825958e9ull, 0x3b2fe622407e315bull, 0xcad7986f4654af33ull, 0xe460e73fe1e27cf0ull, 
0xb9344703756cfaefull, 0xaa40770ab31c830aull, 0xa880f6e28e65d039ull, 0xaec317811926c7e9ull, 
0xf150ba79062bea34ull, 0xd48e58f3116a4268ull, 0x9ea815f597e35da4ull, 0x7039b9e61e26c84aull, 
0x3eefdaad53adf2f8ull, 0x5f3d793b1deb4166ull, 0x14e47d06be950293ull, 0x62b8734d78e83bebull, 
0xf0b438d454d372beull, 0xaf17f718dd076d37ull, 0x75bd02aed736c479ull, 0x550638db61bc120eull, 
0xb28896c4e96740c1ull, 0xcb740e9ca5306883ull, 0x549e1d1f26957a84ull, 0xe405c3df0b54c052ull, 
0xcc46a7bfe0e355a9ull, 0xde33dfb0d28f0158ull, 0x3ed9f4017c627f3dull, 0xe124c9d8affbd805ull, 
0x8847f58df69647bbull, 0xd1b46f2964cde56cull, 0xb8f386038142df78ull, 0xf32cd5a0d9e0fdc6ull, 
0x5bd9db17065b2e4dull, 0x1048c66644311b84ull, 0x24b12b0fcb931d1full, 0x6947115c0a7067aeull, 
0xeace689ceeeb9bb2ull, 0xbf69352ba805e3daull, 0x8f9ffd46dcf0234dull, 0xe7241b68bf610361ull, 
0xd400ab94780f5c5aull, 0x015523d135c55c98ull, 0x5004e65eb2b37ccdull, 0xf1885c4eb8fff31cull, 
0x11b19b9e732adb95ull, 0x4369f17b9463db52ull, 0x967dd0f4f20c30d0ull, 0x79d46e3e8eb0835cull, 
0xc11191839fa454f8ull, 0xd0c37879f8721cc1ull, 0xb7c25bf2b1ad6ec2ull, 0xe0ca3dbcbccc917cull,
};

const uint64_t zob1[64] = {
0x6a43c083b7a7d552ull, 0x87caad3d937b9b75ull, 0x0462397716b2cd36ull, 0x1341ef131df700ceull, 
0x6765bf7621089a4full, 0x58af8231a2714e0cull, 0x221438f13d217b0bull, 0x2b4e0a20d40d1778ull, 
0x4b9d7ab7790a64ecull, 0x3423f44cbbf22283ull, 0x0850597d42877e2bull, 0x07ac1679f3a8c52cull, 
0x18bcc6a2b6c5762bull, 0xbead75580d155a6bull, 0x434e448b1246ea0dull, 0x18f81a794b6d2418ull, 
0xa6c9c3bb41a0255eull, 0x7ef63ef3e815f981ull, 0xf30c3d5b7190a2efull, 0x4af673f860518ae8ull, 
0x4c4a868c1384bc6dull, 0xcb1947fafc459543ull, 0x34b2a31afd519b48ull, 0x1fc1cb75054c38d2ull, 
0x0224791aa952835full, 0x039acb66c01db264ull, 0x35b9303f964c64c7ull, 0x8d912909a7fcb42eull, 
0xcad5ffcf54b94855ull, 0x14b01f3cb0f6c850ull, 0x1fdc2ea98adc18dbull, 0xae006dd1ee658961ull, 
0x1ead788b67484b39ull, 0xca2dbf66080f0e80ull, 0x15b443db11f1b388ull, 0x90f7bb2f05cb3689ull, 
0x25fb04f12f5b4101ull, 0x93644e43fae4eb94ull, 0x55dd9ec6fe076abcull, 0x1bdee2d00abc4565ull, 
0x7e48a2bd4a3e5adaull, 0xb6842e93dd8595e6ull, 0x65b90fa62a9673e6ull, 0x52fc814f519c3e1bull, 
0x2e25123c733da701ull, 0x289a056664bf598full, 0xa59c30f1eb690b07ull, 0x447d59acd69cf651ull, 
0x00c0dd78c3714e32ull, 0x13417089a59ede0eull, 0x1fdb3e4a129b6368ull, 0x76e99a0cad62c9f1ull, 
0x139f5b641a8b463dull, 0x292b118ed8ce881aull, 0xa6de00a17f781120ull, 0x026f8bc108d64edeull, 
0x6b462d684bc4d9dbull, 0xf3bcfee3f4692ec6ull, 0x57df6d044ae10effull, 0x3aef7658da36d76aull, 
0xd1f0311d2b5e403cull, 0x510b773fe3223973ull, 0xb22aba50f53fa7bcull, 0x7df1caec709f35b8ull,
};

template<int C> uint64_t hashPos(int p);
template<> uint64_t hashPos<BLACK>(int p) { return zob0[p]; }
template<> uint64_t hashPos<WHITE>(int p) { return zob1[p]; }

uint64_t hashKo(int p) { return hashPos<BLACK>(p) ^ hashPos<WHITE>(p); }
uint64_t hashChangeSide() { return hashPos<BLACK>(0); }

