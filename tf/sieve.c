typedef unsigned long long u64;
void sieve03(u64 *p, u64 *end) { do {
p[0] = 0x9249249249249249;
p[1] = 0x4924924924924924;
p[2] = 0x2492492492492492;
p += 3;
} while (p < end); }
void sieve05(u64 *p, u64 *end) { do {
p[0] |= 0x1084210842108421;
p[1] |= 0x2108421084210842;
p[2] |= 0x4210842108421084;
p[3] |= 0x8421084210842108;
p[4] |= 0x0842108421084210;
p += 5;
} while (p < end); }
void sieve07(u64 *p, u64 *end) { do {
p[0] |= 0x8102040810204081;
p[1] |= 0x4081020408102040;
p[2] |= 0x2040810204081020;
p[3] |= 0x1020408102040810;
p[4] |= 0x0810204081020408;
p[5] |= 0x0408102040810204;
p[6] |= 0x0204081020408102;
p += 7;
} while (p < end); }
void sieve11(u64 *p, u64 *end) { do {
p[0] |= 0x0080100200400801;
p[1] |= 0x0200400801002004;
p[2] |= 0x0801002004008010;
p[3] |= 0x2004008010020040;
p[4] |= 0x8010020040080100;
p[5] |= 0x0040080100200400;
p[6] |= 0x0100200400801002;
p[7] |= 0x0400801002004008;
p[8] |= 0x1002004008010020;
p[9] |= 0x4008010020040080;
p[10] |= 0x0020040080100200;
p += 11;
} while (p < end); }
void sieve13(u64 *p, u64 *end) { do {
p[0] |= 0x0010008004002001;
p[1] |= 0x0020010008004002;
p[2] |= 0x0040020010008004;
p[3] |= 0x0080040020010008;
p[4] |= 0x0100080040020010;
p[5] |= 0x0200100080040020;
p[6] |= 0x0400200100080040;
p[7] |= 0x0800400200100080;
p[8] |= 0x1000800400200100;
p[9] |= 0x2001000800400200;
p[10] |= 0x4002001000800400;
p[11] |= 0x8004002001000800;
p[12] |= 0x0008004002001000;
p += 13;
} while (p < end); }
void sieve17(u64 *p, u64 *end) { do {
p[0] |= 0x0008000400020001;
p[1] |= 0x0080004000200010;
p[2] |= 0x0800040002000100;
p[3] |= 0x8000400020001000;
p[4] |= 0x0004000200010000;
p[5] |= 0x0040002000100008;
p[6] |= 0x0400020001000080;
p[7] |= 0x4000200010000800;
p[8] |= 0x0002000100008000;
p[9] |= 0x0020001000080004;
p[10] |= 0x0200010000800040;
p[11] |= 0x2000100008000400;
p[12] |= 0x0001000080004000;
p[13] |= 0x0010000800040002;
p[14] |= 0x0100008000400020;
p[15] |= 0x1000080004000200;
p[16] |= 0x0000800040002000;
p += 17;
} while (p < end); }
void sieve19(u64 *p, u64 *end) { do {
p[0] |= 0x0200004000080001;
p[1] |= 0x0004000080001000;
p[2] |= 0x4000080001000020;
p[3] |= 0x0080001000020000;
p[4] |= 0x0001000020000400;
p[5] |= 0x1000020000400008;
p[6] |= 0x0020000400008000;
p[7] |= 0x0000400008000100;
p[8] |= 0x0400008000100002;
p[9] |= 0x0008000100002000;
p[10] |= 0x8000100002000040;
p[11] |= 0x0100002000040000;
p[12] |= 0x0002000040000800;
p[13] |= 0x2000040000800010;
p[14] |= 0x0040000800010000;
p[15] |= 0x0000800010000200;
p[16] |= 0x0800010000200004;
p[17] |= 0x0010000200004000;
p[18] |= 0x0000200004000080;
p += 19;
} while (p < end); }
void sieve23(u64 *p, u64 *end) { do {
p[0] |= 0x0000400000800001;
p[1] |= 0x0008000010000020;
p[2] |= 0x0100000200000400;
p[3] |= 0x2000004000008000;
p[4] |= 0x0000080000100000;
p[5] |= 0x0001000002000004;
p[6] |= 0x0020000040000080;
p[7] |= 0x0400000800001000;
p[8] |= 0x8000010000020000;
p[9] |= 0x0000200000400000;
p[10] |= 0x0004000008000010;
p[11] |= 0x0080000100000200;
p[12] |= 0x1000002000004000;
p[13] |= 0x0000040000080000;
p[14] |= 0x0000800001000002;
p[15] |= 0x0010000020000040;
p[16] |= 0x0200000400000800;
p[17] |= 0x4000008000010000;
p[18] |= 0x0000100000200000;
p[19] |= 0x0002000004000008;
p[20] |= 0x0040000080000100;
p[21] |= 0x0800001000002000;
p[22] |= 0x0000020000040000;
p += 23;
} while (p < end); }
void sieve29(u64 *p, u64 *end) { do {
p[0] |= 0x0400000020000001;
p[1] |= 0x0010000000800000;
p[2] |= 0x0000400000020000;
p[3] |= 0x0000010000000800;
p[4] |= 0x8000000400000020;
p[5] |= 0x0200000010000000;
p[6] |= 0x0008000000400000;
p[7] |= 0x0000200000010000;
p[8] |= 0x0000008000000400;
p[9] |= 0x4000000200000010;
p[10] |= 0x0100000008000000;
p[11] |= 0x0004000000200000;
p[12] |= 0x0000100000008000;
p[13] |= 0x0000004000000200;
p[14] |= 0x2000000100000008;
p[15] |= 0x0080000004000000;
p[16] |= 0x0002000000100000;
p[17] |= 0x0000080000004000;
p[18] |= 0x0000002000000100;
p[19] |= 0x1000000080000004;
p[20] |= 0x0040000002000000;
p[21] |= 0x0001000000080000;
p[22] |= 0x0000040000002000;
p[23] |= 0x0000001000000080;
p[24] |= 0x0800000040000002;
p[25] |= 0x0020000001000000;
p[26] |= 0x0000800000040000;
p[27] |= 0x0000020000001000;
p[28] |= 0x0000000800000040;
p += 29;
} while (p < end); }
void sieve31(u64 *p, u64 *end) { do {
p[0] |= 0x4000000080000001;
p[1] |= 0x1000000020000000;
p[2] |= 0x0400000008000000;
p[3] |= 0x0100000002000000;
p[4] |= 0x0040000000800000;
p[5] |= 0x0010000000200000;
p[6] |= 0x0004000000080000;
p[7] |= 0x0001000000020000;
p[8] |= 0x0000400000008000;
p[9] |= 0x0000100000002000;
p[10] |= 0x0000040000000800;
p[11] |= 0x0000010000000200;
p[12] |= 0x0000004000000080;
p[13] |= 0x0000001000000020;
p[14] |= 0x0000000400000008;
p[15] |= 0x8000000100000002;
p[16] |= 0x2000000040000000;
p[17] |= 0x0800000010000000;
p[18] |= 0x0200000004000000;
p[19] |= 0x0080000001000000;
p[20] |= 0x0020000000400000;
p[21] |= 0x0008000000100000;
p[22] |= 0x0002000000040000;
p[23] |= 0x0000800000010000;
p[24] |= 0x0000200000004000;
p[25] |= 0x0000080000001000;
p[26] |= 0x0000020000000400;
p[27] |= 0x0000008000000100;
p[28] |= 0x0000002000000040;
p[29] |= 0x0000000800000010;
p[30] |= 0x0000000200000004;
p += 31;
} while (p < end); }
void sieve37(u64 *p, u64 *end) { do {
p[0] |= 0x0000002000000001;
p[1] |= 0x0000800000000400;
p[2] |= 0x0200000000100000;
p[3] |= 0x0000000040000000;
p[4] |= 0x0000010000000008;
p[5] |= 0x0004000000002000;
p[6] |= 0x1000000000800000;
p[7] |= 0x0000000200000000;
p[8] |= 0x0000080000000040;
p[9] |= 0x0020000000010000;
p[10] |= 0x8000000004000000;
p[11] |= 0x0000001000000000;
p[12] |= 0x0000400000000200;
p[13] |= 0x0100000000080000;
p[14] |= 0x0000000020000000;
p[15] |= 0x0000008000000004;
p[16] |= 0x0002000000001000;
p[17] |= 0x0800000000400000;
p[18] |= 0x0000000100000000;
p[19] |= 0x0000040000000020;
p[20] |= 0x0010000000008000;
p[21] |= 0x4000000002000000;
p[22] |= 0x0000000800000000;
p[23] |= 0x0000200000000100;
p[24] |= 0x0080000000040000;
p[25] |= 0x0000000010000000;
p[26] |= 0x0000004000000002;
p[27] |= 0x0001000000000800;
p[28] |= 0x0400000000200000;
p[29] |= 0x0000000080000000;
p[30] |= 0x0000020000000010;
p[31] |= 0x0008000000004000;
p[32] |= 0x2000000001000000;
p[33] |= 0x0000000400000000;
p[34] |= 0x0000100000000080;
p[35] |= 0x0040000000020000;
p[36] |= 0x0000000008000000;
p += 37;
} while (p < end); }
void sieve41(u64 *p, u64 *end) { do {
p[0] |= 0x0000020000000001;
p[1] |= 0x0800000000040000;
p[2] |= 0x0000001000000000;
p[3] |= 0x0040000000002000;
p[4] |= 0x0000000080000000;
p[5] |= 0x0002000000000100;
p[6] |= 0x0000000004000000;
p[7] |= 0x0000100000000008;
p[8] |= 0x4000000000200000;
p[9] |= 0x0000008000000000;
p[10] |= 0x0200000000010000;
p[11] |= 0x0000000400000000;
p[12] |= 0x0010000000000800;
p[13] |= 0x0000000020000000;
p[14] |= 0x0000800000000040;
p[15] |= 0x0000000001000000;
p[16] |= 0x0000040000000002;
p[17] |= 0x1000000000080000;
p[18] |= 0x0000002000000000;
p[19] |= 0x0080000000004000;
p[20] |= 0x0000000100000000;
p[21] |= 0x0004000000000200;
p[22] |= 0x0000000008000000;
p[23] |= 0x0000200000000010;
p[24] |= 0x8000000000400000;
p[25] |= 0x0000010000000000;
p[26] |= 0x0400000000020000;
p[27] |= 0x0000000800000000;
p[28] |= 0x0020000000001000;
p[29] |= 0x0000000040000000;
p[30] |= 0x0001000000000080;
p[31] |= 0x0000000002000000;
p[32] |= 0x0000080000000004;
p[33] |= 0x2000000000100000;
p[34] |= 0x0000004000000000;
p[35] |= 0x0100000000008000;
p[36] |= 0x0000000200000000;
p[37] |= 0x0008000000000400;
p[38] |= 0x0000000010000000;
p[39] |= 0x0000400000000020;
p[40] |= 0x0000000000800000;
p += 41;
} while (p < end); }
void sieve43(u64 *p, u64 *end) { do {
p[0] |= 0x0000080000000001;
p[1] |= 0x0000000000400000;
p[2] |= 0x0000100000000002;
p[3] |= 0x0000000000800000;
p[4] |= 0x0000200000000004;
p[5] |= 0x0000000001000000;
p[6] |= 0x0000400000000008;
p[7] |= 0x0000000002000000;
p[8] |= 0x0000800000000010;
p[9] |= 0x0000000004000000;
p[10] |= 0x0001000000000020;
p[11] |= 0x0000000008000000;
p[12] |= 0x0002000000000040;
p[13] |= 0x0000000010000000;
p[14] |= 0x0004000000000080;
p[15] |= 0x0000000020000000;
p[16] |= 0x0008000000000100;
p[17] |= 0x0000000040000000;
p[18] |= 0x0010000000000200;
p[19] |= 0x0000000080000000;
p[20] |= 0x0020000000000400;
p[21] |= 0x0000000100000000;
p[22] |= 0x0040000000000800;
p[23] |= 0x0000000200000000;
p[24] |= 0x0080000000001000;
p[25] |= 0x0000000400000000;
p[26] |= 0x0100000000002000;
p[27] |= 0x0000000800000000;
p[28] |= 0x0200000000004000;
p[29] |= 0x0000001000000000;
p[30] |= 0x0400000000008000;
p[31] |= 0x0000002000000000;
p[32] |= 0x0800000000010000;
p[33] |= 0x0000004000000000;
p[34] |= 0x1000000000020000;
p[35] |= 0x0000008000000000;
p[36] |= 0x2000000000040000;
p[37] |= 0x0000010000000000;
p[38] |= 0x4000000000080000;
p[39] |= 0x0000020000000000;
p[40] |= 0x8000000000100000;
p[41] |= 0x0000040000000000;
p[42] |= 0x0000000000200000;
p += 43;
} while (p < end); }
void sieve47(u64 *p, u64 *end) { do {
p[0] |= 0x0000800000000001;
p[1] |= 0x0000000040000000;
p[2] |= 0x1000000000002000;
p[3] |= 0x0000080000000000;
p[4] |= 0x0000000004000000;
p[5] |= 0x0100000000000200;
p[6] |= 0x0000008000000000;
p[7] |= 0x0000000000400000;
p[8] |= 0x0010000000000020;
p[9] |= 0x0000000800000000;
p[10] |= 0x0000000000040000;
p[11] |= 0x0001000000000002;
p[12] |= 0x0000000080000000;
p[13] |= 0x2000000000004000;
p[14] |= 0x0000100000000000;
p[15] |= 0x0000000008000000;
p[16] |= 0x0200000000000400;
p[17] |= 0x0000010000000000;
p[18] |= 0x0000000000800000;
p[19] |= 0x0020000000000040;
p[20] |= 0x0000001000000000;
p[21] |= 0x0000000000080000;
p[22] |= 0x0002000000000004;
p[23] |= 0x0000000100000000;
p[24] |= 0x4000000000008000;
p[25] |= 0x0000200000000000;
p[26] |= 0x0000000010000000;
p[27] |= 0x0400000000000800;
p[28] |= 0x0000020000000000;
p[29] |= 0x0000000001000000;
p[30] |= 0x0040000000000080;
p[31] |= 0x0000002000000000;
p[32] |= 0x0000000000100000;
p[33] |= 0x0004000000000008;
p[34] |= 0x0000000200000000;
p[35] |= 0x8000000000010000;
p[36] |= 0x0000400000000000;
p[37] |= 0x0000000020000000;
p[38] |= 0x0800000000001000;
p[39] |= 0x0000040000000000;
p[40] |= 0x0000000002000000;
p[41] |= 0x0080000000000100;
p[42] |= 0x0000004000000000;
p[43] |= 0x0000000000200000;
p[44] |= 0x0008000000000010;
p[45] |= 0x0000000400000000;
p[46] |= 0x0000000000020000;
p += 47;
} while (p < end); }
void sieve53(u64 *p, u64 *end) { do {
p[0] |= 0x0020000000000001;
p[1] |= 0x0000040000000000;
p[2] |= 0x0000000080000000;
p[3] |= 0x0000000000100000;
p[4] |= 0x4000000000000200;
p[5] |= 0x0008000000000000;
p[6] |= 0x0000010000000000;
p[7] |= 0x0000000020000000;
p[8] |= 0x0000000000040000;
p[9] |= 0x1000000000000080;
p[10] |= 0x0002000000000000;
p[11] |= 0x0000004000000000;
p[12] |= 0x0000000008000000;
p[13] |= 0x0000000000010000;
p[14] |= 0x0400000000000020;
p[15] |= 0x0000800000000000;
p[16] |= 0x0000001000000000;
p[17] |= 0x0000000002000000;
p[18] |= 0x0000000000004000;
p[19] |= 0x0100000000000008;
p[20] |= 0x0000200000000000;
p[21] |= 0x0000000400000000;
p[22] |= 0x0000000000800000;
p[23] |= 0x0000000000001000;
p[24] |= 0x0040000000000002;
p[25] |= 0x0000080000000000;
p[26] |= 0x0000000100000000;
p[27] |= 0x0000000000200000;
p[28] |= 0x8000000000000400;
p[29] |= 0x0010000000000000;
p[30] |= 0x0000020000000000;
p[31] |= 0x0000000040000000;
p[32] |= 0x0000000000080000;
p[33] |= 0x2000000000000100;
p[34] |= 0x0004000000000000;
p[35] |= 0x0000008000000000;
p[36] |= 0x0000000010000000;
p[37] |= 0x0000000000020000;
p[38] |= 0x0800000000000040;
p[39] |= 0x0001000000000000;
p[40] |= 0x0000002000000000;
p[41] |= 0x0000000004000000;
p[42] |= 0x0000000000008000;
p[43] |= 0x0200000000000010;
p[44] |= 0x0000400000000000;
p[45] |= 0x0000000800000000;
p[46] |= 0x0000000001000000;
p[47] |= 0x0000000000002000;
p[48] |= 0x0080000000000004;
p[49] |= 0x0000100000000000;
p[50] |= 0x0000000200000000;
p[51] |= 0x0000000000400000;
p[52] |= 0x0000000000000800;
p += 53;
} while (p < end); }
void sieve59(u64 *p, u64 *end) { do {
p[0] |= 0x0800000000000001;
p[1] |= 0x0040000000000000;
p[2] |= 0x0002000000000000;
p[3] |= 0x0000100000000000;
p[4] |= 0x0000008000000000;
p[5] |= 0x0000000400000000;
p[6] |= 0x0000000020000000;
p[7] |= 0x0000000001000000;
p[8] |= 0x0000000000080000;
p[9] |= 0x0000000000004000;
p[10] |= 0x0000000000000200;
p[11] |= 0x8000000000000010;
p[12] |= 0x0400000000000000;
p[13] |= 0x0020000000000000;
p[14] |= 0x0001000000000000;
p[15] |= 0x0000080000000000;
p[16] |= 0x0000004000000000;
p[17] |= 0x0000000200000000;
p[18] |= 0x0000000010000000;
p[19] |= 0x0000000000800000;
p[20] |= 0x0000000000040000;
p[21] |= 0x0000000000002000;
p[22] |= 0x0000000000000100;
p[23] |= 0x4000000000000008;
p[24] |= 0x0200000000000000;
p[25] |= 0x0010000000000000;
p[26] |= 0x0000800000000000;
p[27] |= 0x0000040000000000;
p[28] |= 0x0000002000000000;
p[29] |= 0x0000000100000000;
p[30] |= 0x0000000008000000;
p[31] |= 0x0000000000400000;
p[32] |= 0x0000000000020000;
p[33] |= 0x0000000000001000;
p[34] |= 0x0000000000000080;
p[35] |= 0x2000000000000004;
p[36] |= 0x0100000000000000;
p[37] |= 0x0008000000000000;
p[38] |= 0x0000400000000000;
p[39] |= 0x0000020000000000;
p[40] |= 0x0000001000000000;
p[41] |= 0x0000000080000000;
p[42] |= 0x0000000004000000;
p[43] |= 0x0000000000200000;
p[44] |= 0x0000000000010000;
p[45] |= 0x0000000000000800;
p[46] |= 0x0000000000000040;
p[47] |= 0x1000000000000002;
p[48] |= 0x0080000000000000;
p[49] |= 0x0004000000000000;
p[50] |= 0x0000200000000000;
p[51] |= 0x0000010000000000;
p[52] |= 0x0000000800000000;
p[53] |= 0x0000000040000000;
p[54] |= 0x0000000002000000;
p[55] |= 0x0000000000100000;
p[56] |= 0x0000000000008000;
p[57] |= 0x0000000000000400;
p[58] |= 0x0000000000000020;
p += 59;
} while (p < end); }
void sieve61(u64 *p, u64 *end) { do {
p[0] |= 0x2000000000000001;
p[1] |= 0x0400000000000000;
p[2] |= 0x0080000000000000;
p[3] |= 0x0010000000000000;
p[4] |= 0x0002000000000000;
p[5] |= 0x0000400000000000;
p[6] |= 0x0000080000000000;
p[7] |= 0x0000010000000000;
p[8] |= 0x0000002000000000;
p[9] |= 0x0000000400000000;
p[10] |= 0x0000000080000000;
p[11] |= 0x0000000010000000;
p[12] |= 0x0000000002000000;
p[13] |= 0x0000000000400000;
p[14] |= 0x0000000000080000;
p[15] |= 0x0000000000010000;
p[16] |= 0x0000000000002000;
p[17] |= 0x0000000000000400;
p[18] |= 0x0000000000000080;
p[19] |= 0x0000000000000010;
p[20] |= 0x4000000000000002;
p[21] |= 0x0800000000000000;
p[22] |= 0x0100000000000000;
p[23] |= 0x0020000000000000;
p[24] |= 0x0004000000000000;
p[25] |= 0x0000800000000000;
p[26] |= 0x0000100000000000;
p[27] |= 0x0000020000000000;
p[28] |= 0x0000004000000000;
p[29] |= 0x0000000800000000;
p[30] |= 0x0000000100000000;
p[31] |= 0x0000000020000000;
p[32] |= 0x0000000004000000;
p[33] |= 0x0000000000800000;
p[34] |= 0x0000000000100000;
p[35] |= 0x0000000000020000;
p[36] |= 0x0000000000004000;
p[37] |= 0x0000000000000800;
p[38] |= 0x0000000000000100;
p[39] |= 0x0000000000000020;
p[40] |= 0x8000000000000004;
p[41] |= 0x1000000000000000;
p[42] |= 0x0200000000000000;
p[43] |= 0x0040000000000000;
p[44] |= 0x0008000000000000;
p[45] |= 0x0001000000000000;
p[46] |= 0x0000200000000000;
p[47] |= 0x0000040000000000;
p[48] |= 0x0000008000000000;
p[49] |= 0x0000001000000000;
p[50] |= 0x0000000200000000;
p[51] |= 0x0000000040000000;
p[52] |= 0x0000000008000000;
p[53] |= 0x0000000001000000;
p[54] |= 0x0000000000200000;
p[55] |= 0x0000000000040000;
p[56] |= 0x0000000000008000;
p[57] |= 0x0000000000001000;
p[58] |= 0x0000000000000200;
p[59] |= 0x0000000000000040;
p[60] |= 0x0000000000000008;
p += 61;
} while (p < end); }
void sieve(int n, u64 *p, u64 *end) {
switch (n) {
case 03: sieve03(p, end); break;
case 05: sieve05(p, end); break;
case 07: sieve07(p, end); break;
case 11: sieve11(p, end); break;
case 13: sieve13(p, end); break;
case 17: sieve17(p, end); break;
case 19: sieve19(p, end); break;
case 23: sieve23(p, end); break;
case 29: sieve29(p, end); break;
case 31: sieve31(p, end); break;
case 37: sieve37(p, end); break;
case 41: sieve41(p, end); break;
case 43: sieve43(p, end); break;
case 47: sieve47(p, end); break;
case 53: sieve53(p, end); break;
case 59: sieve59(p, end); break;
case 61: sieve61(p, end); break;
}}
