#include <sys/time.h>

unsigned long timeMillis() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

void time(const char *s = 0) {
  static unsigned long prev = 0;
  unsigned long now = timeMillis();
  if (prev && s) {
    printf("%s: %lu ms\n", s, now - prev);
  }
  prev = now;
}
