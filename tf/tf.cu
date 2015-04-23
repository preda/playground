__device__ void square(uint64_t a, uint64_t b, ) {
  asm("{\n"
      "shl.u64        %2, %4, 1;"
      "mul.lo.u64     %0, %3, %3;"
      "mul.hi.u64     %1, %3, %3;"
      "mad.lo.cc.u64  %1, %3, %2, %1;"
      "mulc.hi.cc.u64 %2, %3, %2;"
      "madc.lo.u64    %2, %4, %4, %2;"
      "\n}");
}

/*
      "mul.lo.u64     %1, %3, %2;"
      "madc.hi.cc.u64 %1, %3, %3, %1;"
      "madc.hi.u64    %2, %3, %2;"
      "mad.lo.u64     %2, %4, %4, %2;"
      
      "mul.hi.u64 %1, %3, %3;"
      "mul.lo.u64 %2, %3, %4;"

      "mul.lo.u64 %0, %3, %3;"
*/
    
