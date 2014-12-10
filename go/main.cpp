#include "go.h"
#include "gtp.h"

#include <cstdio>

std::vector<StringCommand> stringCommands = {
  {"protocol_version", "2"},
  {"name", "microgo"},
  {"version", "1"},
  {"quit", ""}
};

std::vector<gtp_command> commands = {
  // {"boardsize", [](char *) { }},
};

int main() {
  gtp_main_loop(stringCommands, commands, stdin, stdout, stderr);

  /*
  Board b;
  b.move(pos(1, 1), BLACK);
  b.move(pos(0, 1), BLACK);
  auto black = b.region(pos(1, 1), Black);
  print(black);
  auto libs = border(black, Empty);
  print(libs);
  printf("Dead %d\n", isDead(black));
  */
}
