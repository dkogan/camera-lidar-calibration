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

    int Nfailed = 0;
    if(0 != memcmp(ref, bitarray, Nwords*sizeof(uint64_t)))
    {
        printf("\x1b[31m"
               "FAILED: mismatched data"
               "\x1b[0m\n");
        Nfailed++;
    }

    if(bitarray64_check(bitarray,64*2+50-1))
    {
        printf("\x1b[31m"
               "FAILED: bit 64*2+50-1 should be clear"
               "\x1b[0m\n");
        Nfailed++;
    }
    if(!bitarray64_check(bitarray,64*2+50))
    {
        printf("\x1b[31m"
               "FAILED: bit 64*2+50 should be set"
               "\x1b[0m\n");
        Nfailed++;
    }
    if(!bitarray64_check(bitarray,64*2+50+100-1))
    {
        printf("\x1b[31m"
               "FAILED: bit 64*2+50+100-1 should be set"
               "\x1b[0m\n");
        Nfailed++;
    }
    if(bitarray64_check(bitarray,64*2+50+100))
    {
        printf("\x1b[31m"
               "FAILED: bit 64*2+50+100 should be clear"
               "\x1b[0m\n");
        Nfailed++;
    }

    if(bitarray64_check_all_set(bitarray,Nbits))
    {
        printf("\x1b[31m"
               "FAILED: should NOT be all_set"
               "\x1b[0m\n");
        Nfailed++;
    }
    if(bitarray64_check_all_clear(bitarray,Nbits))
    {
        printf("\x1b[31m"
               "FAILED: should NOT be all_clear"
               "\x1b[0m\n");
        Nfailed++;
    }


    for(int i=0; i<Nwords; i++) bitarray[i] = 0UL;
    if(bitarray64_check_all_set(bitarray,Nbits))
    {
        printf("\x1b[31m"
               "FAILED: should NOT be all_set"
               "\x1b[0m\n");
        Nfailed++;
    }
    if(!bitarray64_check_all_clear(bitarray,Nbits))
    {
        printf("\x1b[31m"
               "FAILED: should be all_clear"
               "\x1b[0m\n");
        Nfailed++;
    }

    // Set one-bit-past-the end. This is out-of-bounds and we should still be
    // all clear
    bitarray[Nbits/64] |= 1UL << (Nbits%64);
    if(bitarray64_check_all_set(bitarray,Nbits))
    {
        printf("\x1b[31m"
               "FAILED: should NOT be all_set"
               "\x1b[0m\n");
        Nfailed++;
    }
    if(!bitarray64_check_all_clear(bitarray,Nbits))
    {
        printf("\x1b[31m"
               "FAILED: should be all_clear"
               "\x1b[0m\n");
        Nfailed++;
    }

    // Set the last bit
    bitarray[Nbits/64] |= 1UL << ((Nbits%64)-1);
    if(bitarray64_check_all_set(bitarray,Nbits))
    {
        printf("\x1b[31m"
               "FAILED: should NOT be all_set"
               "\x1b[0m\n");
        Nfailed++;
    }
    if(bitarray64_check_all_clear(bitarray,Nbits))
    {
        printf("\x1b[31m"
               "FAILED: should NOT be all_clear"
               "\x1b[0m\n");
        Nfailed++;
    }
    bitarray[Nbits/64] = 0;
    if(!bitarray64_check_all_clear(bitarray,Nbits))
    {
        printf("\x1b[31m"
               "FAILED: should be all_clear"
               "\x1b[0m\n");
        Nfailed++;
    }

    // Set the first bit
    bitarray[0] |= 1UL;
    if(bitarray64_check_all_set(bitarray,Nbits))
    {
        printf("\x1b[31m"
               "FAILED: should NOT be all_set"
               "\x1b[0m\n");
        Nfailed++;
    }
    if(bitarray64_check_all_clear(bitarray,Nbits))
    {
        printf("\x1b[31m"
               "FAILED: should NOT be all_clear"
               "\x1b[0m\n");
        Nfailed++;
    }



    if(Nfailed == 0)
    {
        printf("\x1b[32m"
               "ALL OK"
               "\x1b[0m\n");
        return 0;
    }

    printf("\x1b[31m"
           "%d tests failed"
           "\x1b[0m\n", Nfailed);

    return Nfailed;
}
