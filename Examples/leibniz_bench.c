#include <stdio.h>
#include <stdlib.h>
unsigned rounds; double pi = 1.0;
int main() {
  FILE *infile = fopen("rounds.txt", "r");
  fscanf(infile, "%u", &rounds); fclose(infile);
  rounds += 2u;
  for (unsigned i=2u; i < rounds; ++i) {
    double x = -1.0 + 2.0 * (i & 0x1);
    pi += (x / (2u * i - 1u));
  }
  pi *= 4; printf("%.16f\n", pi); return 0;
}
