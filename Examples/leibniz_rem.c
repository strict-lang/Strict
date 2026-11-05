#include <stdio.h>
int main() {
  unsigned rounds = 1000000002u;
  double pi = 1.0;
  for (unsigned i = 2u; i < rounds; ++i) {
    long long x = -1 + 2 * (i % 2);
    pi += (double)x / (double)(2u * i - 1u);
  }
  printf("%.16f\n", pi * 4);
}
