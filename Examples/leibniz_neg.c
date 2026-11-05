#include <stdio.h>
#include <stdlib.h>
int main() {
  unsigned rounds; double pi = 1.0; double x = 1.0;
  FILE *infile = fopen("rounds.txt", "r");
  fscanf(infile, "%u", &rounds); fclose(infile);
  rounds += 2u;
  for (unsigned i=2u; i < rounds; ++i) { x = -x; pi += x / (2u * i - 1u); }
  printf("%.16f\n", pi * 4);
}
