#pragma once

/* bitarray. Test it like this:

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
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

    for(int i=0; i<Nwords; i++)
    {
        printf("word %d ref/computed/xor:\n0x%016"PRIx64"\n0x%016"PRIx64"\n0x%016"PRIx64"\n\n",
               i,
               ref[i],
               bitarray[i],
               ref[i] ^ bitarray[i]);
    }
    if(0 != memcmp(ref, bitarray, Nwords*sizeof(uint64_t)))
    {
        printf("Mismatched data\n");
        return 1;
    }
    printf("All OK\n");
    return 0;
}

*/
__attribute__((unused))
static inline int bitarray64_nwords(const int Nbits)
{
    // round up the number of 64-bit words required
    return (Nbits+63)/64;
}
__attribute__((unused))
static inline void bitarray64_set(uint64_t* bitarray, int ibit)
{
    bitarray[ibit/64] |= (1ul << (ibit % 64));
}
__attribute__((unused))
static inline void bitarray64_set_all(uint64_t* bitarray, const int Nwords)
{
    for(int i=0; i<Nwords; i++)
        bitarray[i] = ~0ULL;
}
__attribute__((unused))
static inline void bitarray64_clear_all(uint64_t* bitarray, const int Nwords)
{
    for(int i=0; i<Nwords; i++)
        bitarray[i] = 0ULL;
}
__attribute__((unused))
static inline void bitarray64_clear(uint64_t* bitarray, int ibit)
{
    bitarray[ibit/64] &= ~(1ul << (ibit % 64));
}
__attribute__((unused))
static inline bool bitarray64_check(const uint64_t* bitarray, int ibit)
{
    return bitarray[ibit/64] & (1ul << (ibit % 64));
}
__attribute__((unused))
static inline void bitarray64_set_range_oneword(uint64_t* word,
                                                int ibit0, int Nbits)
{
    *word |= ((1ul << Nbits) - 1) << ibit0;
}
__attribute__((unused))
static inline void bitarray64_set_range(uint64_t* bitarray,
                                        int ibit0, int Nbits)
{
    // The first chunk, up to the first word boundary
    int ibit_next_start_of_word = 64*(int)((ibit0+63)/64);

    int Nbits_remaining_in_word = ibit_next_start_of_word - ibit0;
    if(Nbits_remaining_in_word)
    {
        if(Nbits <= Nbits_remaining_in_word)
        {
            bitarray64_set_range_oneword(&bitarray[ibit0/64],
                                         ibit0%64,
                                         Nbits);
            return;
        }

        bitarray64_set_range_oneword(&bitarray[ibit0/64],
                                     ibit0%64,
                                     Nbits_remaining_in_word);

        ibit0 = ibit_next_start_of_word;
        Nbits -= Nbits_remaining_in_word;
    }

    // Next chunk starts at an even word boundary

    // Process any whole words
    while(Nbits >= 64)
    {
        bitarray[ibit0/64] = ~0ul;
        ibit0 += 64;
        Nbits -= 64;
    }

    // Last little bit
    if(Nbits)
        bitarray64_set_range_oneword(&bitarray[ibit0/64],
                                     0,
                                     Nbits);
}
