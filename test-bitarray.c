#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#include "bitarray.h"

int main(int argc      __attribute__((unused)),
         char* argv[]  __attribute__((unused)))
{
    const int Nbits = 350;

    const int Nwords = bitarray64_nwords(Nbits);
    uint64_t bitarray[Nwords];

    if(Nwords != 6)
    {
        printf("Mismatched Nwords\n");
        return 1;
    }
    uint64_t ref[6] = {};

    memset(bitarray, 0, Nwords*sizeof(uint64_t));

    bitarray64_set(      bitarray, 1);
    bitarray64_set_range(bitarray, 5, 30);
    bitarray64_clear(    bitarray, 6);
    bitarray64_set_range(bitarray, 60,4);
    ref[0] = 0xf0000007ffffffa2;

    bitarray64_set_range(bitarray, 64*1 + 60, 7);
    ref[1] = 0xf000000000000000;

    bitarray64_set_range(bitarray, 64*2 + 50, 100);
    ref[2] = 0xfffc000000000007;
    ref[3] = 0xffffffffffffffff;
    ref[4] = 0x00000000003fffff;

    bitarray64_set_range(bitarray, 64*5 + 0,  20);
    ref[5] = 0x00000000000fffff;

    if(false)
        for(int i=0; i<Nwords; i++)
            printf("word %d ref/computed/xor:\n0x%016"PRIx64"\n0x%016"PRIx64"\n0x%016"PRIx64"\n\n",
                   i,
                   ref[i],
                   bitarray[i],
                   ref[i] ^ bitarray[i]);

    if(0 != memcmp(ref, bitarray, Nwords*sizeof(uint64_t)))
    {
        printf("\x1b[31m"
               "FAILED: mismatched data"
               "\x1b[0m\n");
        return 1;
    }

    printf("\x1b[32m"
           "ALL OK"
           "\x1b[0m\n");
    return 0;
}
