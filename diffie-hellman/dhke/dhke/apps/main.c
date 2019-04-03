#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int public_base    = 2;
int public_modulus = 23;

int powmod(int g, int x, int p) {
   return fmod(pow(g, x), p);
}

int gen_private_key() {
   srand(time(NULL));
   return rand() % 9 + 2;
}

int gen_public_key(int private_key) {
   return powmod(public_base, private_key, public_modulus);
}

int get_remote_public_key() {
   int remote_key = 0;
   printf("remote key:  ");
   if (scanf("%d", &remote_key) != 1)
      exit(1);
   return remote_key;
}

int get_shared_secret(int private_key, int remote_key) {
   return powmod(remote_key, private_key, public_modulus);
}

int steal_remote_private_key(int remote_key) {
   int trial_prod = public_base;
   int stolen_key = 1;
   while (trial_prod != remote_key) {
      trial_prod *= public_base;
      trial_prod %= public_modulus;
      stolen_key++;
   }
   return stolen_key;
}

int main() {
   printf("\n");

   int private_key = gen_private_key();
   printf("private key: %d\n", private_key);

   int public_key = gen_public_key(private_key);
   printf("public key:  %d\n", public_key);

   int remote_key = get_remote_public_key();

   int shared_key = get_shared_secret(private_key, remote_key);
   printf("shared key:  %d\n", shared_key);

   int stolen_key = steal_remote_private_key(remote_key);
   printf("stolen key:  %d\n", stolen_key);

   return 0;
}
