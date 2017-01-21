#include <hcc/hc.hpp>
#include <stdio.h>
#include <iostream>

typedef unsigned uint;

template<uint W>
double read(const hc::array<double> &in, uint line, uint p) { return in[line * W + p]; }

template<uint W, uint N>
void read(double *u, const hc::array<double> &in, uint line, uint p) [[hc]] {
  for (int i = 0; i < N; ++i) { u[i] = read<W>(in, line, p + W / N * i); }
}

template<uint W>
void write(double u, hc::array<double> &out, uint line, uint p) { out[line * W + p] = u; }

template<uint W>
void writeC(double u, hc::array<double> &out, uint line, uint p) [[hc]] {
  write<W>((p & W) ? -u : u, out, line, p % W);
}

template<uint W, uint N>
void writeC(double *u, hc::array<double> &out, uint line, uint p) [[hc]] {
  for (int i = 0; i < N; ++i) { writeC<W>(u[i], out, line, p + W / N * i); }
}

template<bool isDIF, uint W, uint H, uint round>
hc::completion_future fftStep(const hc::array<double> &in, hc::array<double> &out) {
  constexpr uint radixExp = 3;
  
  static_assert((W & (W - 1)) == 0, "W must be pow2");
  static_assert((H & (H - 1)) == 0, "H must be pow2");
  static_assert(H <= W * 2, "H too big");
  static_assert(W * 2 >= (1 << (round + 1) * radixExp), "invalid round/radix for W");
  static_assert(H     >= (1 << (round + 1) * radixExp), "invalid round/radix for H");

  constexpr uint size = W * H;
  constexpr uint divW = 1 << (radixExp - 1);
  constexpr uint divH = 1 << radixExp;
  constexpr uint threadsPerLine = W >> (radixExp - 1);
  constexpr uint threads = threadsPerLine * (H >> radixExp);
  constexpr uint mr = 1 << (round * radixExp);

  std::cerr << "foo" << '\n';
  
  auto future = hc::parallel_for_each(hc::extent<1>(threads), [&in, &out](hc::index<1> idx)[[hc]] {
      constexpr uint revbin[8] = {0, 4, 2, 6, 1, 5, 3, 7};
      uint id = idx[0];
      // out[id] = 2 * in[id] * in[id];

      uint g = id / threadsPerLine;
      uint p = id % threadsPerLine;
      uint j = g % mr;
      uint r = (g & ~(mr - 1)) << radixExp;
      uint e = j * (W >> ((round + 1) * radixExp - 1)); // (j * W) >> ((round + 1) * radixExp - 1);
      uint line = j + r;

      double u[32];
      for (int i = 0; i < 4; ++i) {
        read<W, 4>(u + 4 * i,       in, line + i * mr, p);
        read<W, 4>(u + 4 * (i + 4), in, line + (i + 4) * mr, p);
        // addSub(u[i], u[i + 4]);
      }

      for (int i = 0; i < 8; ++i) { writeC<W, 4>(u + 4 * i, out, line + i * mr, p + e * revbin[i]); }
      
    });
  std::cerr << "bar" << '\n';
  return future;
}


int main() {
  std::vector<double> vect(8 * 1024 * 1024, 2.0);
  hc::array<double> a(8 * 1024 * 1024, vect.begin());
  hc::array<double> b(8 * 1024 * 1024);

  auto r = fftStep<true, 1024, 256, 0>(a, b);
  hc::accelerator acc;
  auto view = acc.get_default_view();
  view.flush();
  // r.flush();
  r.wait();
  printf("%.3f ms\n", (r.get_end_tick() - r.get_begin_tick()) / 1000000.0);

  /*
  for (int i = 0; i < 2; ++i) {
    auto r = hc::parallel_for_each(hc::extent<1>(8 * 1024 * 1024), [&v](hc::index<1> i)[[hc]] {
        v[i] = (v[i] + 1) * (v[i] - 1);

        // uint ii = i[0];
        // vect[ii] = vect[ii] * vect[ii];
      });
    r.wait();
    std::cout << r.get_end_tick() - r.get_begin_tick() << '\n' << r.get_tick_frequency() << '\n';
  }
  std::cout << vect[100] << '\n';
  */
  
  // hc::accelerator acc;
  
}

/*
  hc::accelerator a;
  std::wcout << a.get_device_path() << std::endl;
  for (auto a : hc::accelerator::get_all()) {
    std::wcout << a.get_device_path() << '\n' << a.get_description() << '\n' << a.get_max_tile_static_size() << '\n'
               << a.is_hsa_accelerator() << '\n' << a.get_profile() << '\n' << a.get_cu_count() << std::endl;
  }


*/
  
/*
template<uint N>
void foo(hc::array<double> &out) {
  constexpr uint M = 2 * N;
  auto r = hc::parallel_for_each(hc::extent<1>(N), [&out](hc::index<1> idx)[[hc]]{ out[idx] = M; });
  r.wait();
  std::cout << r.get_end_tick() - r.get_begin_tick() << '\n' << r.get_tick_frequency() << '\n';
}

template<uint N>
void bar(hc::array<double> &out) {
  constexpr uint n = 2 * N;
  foo<n>(out);
}
*/
