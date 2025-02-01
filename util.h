#pragma once

#include <stdbool.h>

#define MSG(fmt, ...) fprintf(stderr, "%s(%d) %s(): " fmt "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__)

#define MSG_IF_VERBOSE(...) do { if(verbose) { MSG(__VA_ARGS__); } } while(0)

// available to this library, but NOT exported outside
void eig_real_symmetric_3x3( // out
                             double* vsmallest, // the smallest-eigenvalue eigenvector; may be NULL
                             double* vlargest,  // the largest-eigenvalue  eigenvector; may be NULL
                             double* l,         // ALL the eigenvalues, in ascending order
                             // in
                             const double* M, // shape (6,); packed storage; row-first
                             const bool normalize_v );
