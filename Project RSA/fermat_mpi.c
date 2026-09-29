#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <mpi.h>
#include "rsa_common.h"

/* Iterations between MPI_Allreduce checks. Trades detection latency for
   reduced collective overhead. 4096 keeps per-check work under ~10us on
   modern hardware while making Allreduce a negligible fraction of runtime. */
#define CHECK_INTERVAL 4096

/* ---------- Check perfect square ---------- */
int is_perfect_square(uint64_t x, uint64_t *root) {
    uint64_t r = (uint64_t)(sqrt((double)x));
    if (r * r == x) {
        *root = r;
        return 1;
    }
    if ((r + 1) * (r + 1) == x) {
        *root = r + 1;
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Shared RSA modulus (see rsa_common.h) */
    uint64_t p_real = RSA_P;
    uint64_t q_real = RSA_Q;
    uint64_t n = p_real * q_real;

    uint64_t a_start = (uint64_t)ceil(sqrt((double)n));
    uint64_t a, b2, b;
    uint64_t p_found = 0, q_found = 0;

    int local_found = 0;
    double t1 = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        t1 = MPI_Wtime();

    /* Each rank searches a[i] = a_start + rank + i*size (stripe by rank).
       Every CHECK_INTERVAL iterations, do a global check to see if anyone
       has found a factor — this replaces the previous per-iteration
       Allreduce, which dominated runtime. */
    a = a_start + (uint64_t)rank;
    int global_found = 0;
    int iter_since_check = 0;

    while (!global_found) {
        b2 = a * a - n;
        if (is_perfect_square(b2, &b)) {
            p_found = a - b;
            q_found = a + b;
            local_found = 1;
        }

        iter_since_check++;
        if (local_found || iter_since_check >= CHECK_INTERVAL) {
            MPI_Allreduce(&local_found, &global_found, 1, MPI_INT, MPI_LOR,
                          MPI_COMM_WORLD);
            iter_since_check = 0;
        }

        a += (uint64_t)size;
    }

    /* Determine WHICH rank found the answer, then broadcast p,q from that
       rank so rank 0 never prints garbage 0,0. MAXLOC picks the maximum
       flag value (1 beats 0), and among ties the smallest rank. */
    struct { int flag; int rank; } my_pair, winner;
    my_pair.flag = local_found;
    my_pair.rank = rank;
    MPI_Allreduce(&my_pair, &winner, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

    /* Winner rank broadcasts its p,q to everyone */
    MPI_Bcast(&p_found, 1, MPI_UNSIGNED_LONG, winner.rank, MPI_COMM_WORLD);
    MPI_Bcast(&q_found, 1, MPI_UNSIGNED_LONG, winner.rank, MPI_COMM_WORLD);

    if (rank == 0) {
        double t2 = MPI_Wtime();

        printf("MPI Fermat Factorization\n");
        printf("------------------------\n");
        printf("n = %lu\n\n", n);

        printf("Recovered factors:\n");
        printf("p = %lu\n", p_found);
        printf("q = %lu\n", q_found);
        printf("p * q = %lu\n", p_found * q_found);
        printf("Matches n : %s\n\n", (p_found * q_found == n) ? "yes" : "NO");

        printf("Factorization time: %.6f ms\n", (t2 - t1) * 1000.0);
        printf("MPI processes used: %d\n", size);
        printf("Winning rank      : %d\n", winner.rank);
    }

    MPI_Finalize();
    return 0;
}
