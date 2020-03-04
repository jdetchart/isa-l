//
// Created by j.detchart on 04/03/20.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>		// for memset, memcmp
#include <getopt.h>
#include <unistd.h>
#include <pthread.h>
#include "erasure_code.h"
#include "test.h"

#define MMAX 255
#define KMAX 255
#  define TEST_SOURCES 256
#define BAD_MATRIX -1

typedef unsigned char u8;
int m,rr,k;
int sz, l;
int t;

typedef enum instruction_set
{
    INTEL_SSE = 0,
    INTEL_AVX,
    INTEL_AVX2,
    INTEL_AVX512
} instruction_set_t;

instruction_set_t instr = 0;

void parse_args(int argc, char**argv)
{
    int option;
    while ((option = getopt(argc, argv, "k:r:s:l:i:t:a:")) > 0)
    {
        switch (option)
        {
            case 'k':
                k = atoi(optarg);
                break;
            case 'r':
                rr = atoi(optarg);
                break;
            case 's':
                sz = atoi(optarg);
                break;
            case 'l':
                l = atoi(optarg);
                break;
            case 'i':
                if (strncmp(optarg, "sse",3) == 0)
                {
                    instr = INTEL_SSE;
                }
                if (strncmp(optarg, "avx",3) == 0)
                {
                    instr = INTEL_AVX;
                }
                if (strncmp(optarg, "avx2",4) == 0)
                {
                    instr = INTEL_AVX2;
                }
                if (strncmp(optarg, "avx512",6) == 0)
                {
#ifdef HAVE_AS_KNOWS_AVX512
                    instr = INTEL_AVX512;
#else
                    printf("Warning: AVX512 not available. Fallback to AVX2\n");
                    instr = INTEL_AVX2;
#endif
                }
                break;
            case 't':
                t = atoi(optarg);
                break;
        }
    }
}

void print_params()
{
    printf("k:%d\n",k);
    printf("r:%d\n",rr);
    printf("sz:%d\n", sz);
    printf("l:%d\n",l);
    switch(instr)
    {
        case INTEL_SSE:
            printf("i:SSE\n");
            break;
        case INTEL_AVX:
            printf("i:AVX\n");
            break;
        case INTEL_AVX2:
            printf("i:AVX2\n");
            break;
        case INTEL_AVX512:
            printf("i:AVX512\n");
            break;
    }
    printf("t:%d\n", t);
}

typedef struct isa_mt_struct
{
    u8 **k_buffs;
    u8 **r_buffs;
    size_t start;
    int sz;
    int k;
    int m;
    u8* g_tbls;

    pthread_barrier_t* barrier;

    int *stop;

    u8 **copy_kbuffs;

} isa_mt_struct_t;

isa_mt_struct_t* create_mt_encoding_struct(size_t id, int nb_threads, int sz, int k, int m, u8* g_tbls, u8**buffs, u8 ** buffs2, pthread_barrier_t *b, int*stop)
{
    int i;
    isa_mt_struct_t* s = malloc(sizeof(isa_mt_struct_t));
    s->k  = k;
    s->m  = m;
    s->barrier = b;
    s->stop = stop;

    int chunks = sz/nb_threads;
    size_t start = chunks * id; // id =[0..nb_threads[
    s->sz = chunks;

    // keep address for gen mat
    s->g_tbls = g_tbls;

    // copy the buffs
    s->k_buffs = malloc(sizeof(u8*) * (k));
    s->r_buffs = malloc(sizeof(u8*) * (m));

    //shift the buffs
    for (i = 0; i < k; i++)
    {
        s->k_buffs[i] = buffs[i] + start;
    }
    for (i = 0; i < m; i++)
    {
        s->r_buffs[i] = buffs2[i] + start;
    }

    return s;
}

isa_mt_struct_t* create_mt_decoding_struct(size_t id, int nb_threads, int sz, int k, int m, u8* g_tbls, u8**buffs, u8 ** buffs2, pthread_barrier_t *b, int*stop)
{
    int i;
    isa_mt_struct_t* s = malloc(sizeof(isa_mt_struct_t));
    s->k  = k;
    s->m  = m;
    s->barrier = b;
    s->stop = stop;
    int chunks = sz/nb_threads;
    size_t start = chunks * id; // id =[0..nb_threads[
    s->sz = chunks;

    s->start = start;

    // keep address for gen mat
    s->g_tbls = g_tbls;

    // copy the buffs
    s->k_buffs = malloc(sizeof(u8*) * (k));
    s->r_buffs = malloc(sizeof(u8*) * (m));

    //shift the buffs
    s->copy_kbuffs = buffs;

    for (i = 0; i < m; i++)
    {
        s->r_buffs[i] = buffs2[i] + start;
    }

    return s;
}

void* run_encoding_sse(isa_mt_struct_t* s)
{
    int sz = s->sz;
    int k = s->k;
    int m = s->m;
    u8* g_tbls = s->g_tbls;
    u8**k_buffs = s->k_buffs;
    u8**r_buffs = s->r_buffs;
    pthread_barrier_t* barrier = s->barrier;
    pthread_barrier_wait(barrier);
    //printf("thread ready (sse)\n");

    while (1)
    {
        //printf("waiting orders\n");
        pthread_barrier_wait(barrier); // this barrier triggers the code execution
        if (*(s->stop) >0) return NULL;
        //
        ec_encode_data_sse(sz, k, m , g_tbls, k_buffs, r_buffs);

        pthread_barrier_wait(barrier);
    }
    return NULL;
}

void* run_encoding_avx(isa_mt_struct_t* s)
{
    int sz = s->sz;
    int k = s->k;
    int m = s->m;
    u8* g_tbls = s->g_tbls;
    u8**k_buffs = s->k_buffs;
    u8**r_buffs = s->r_buffs;
    pthread_barrier_t* barrier = s->barrier;
    pthread_barrier_wait(barrier);
    //printf("thread ready (avx)\n");

    while (1)
    {
        //printf("waiting orders\n");
        pthread_barrier_wait(barrier); // this barrier triggers the code execution
        if (*(s->stop) >0) return NULL;
        //
        ec_encode_data_avx(sz, k, m , g_tbls, k_buffs, r_buffs);

        pthread_barrier_wait(barrier);
    }
    return NULL;
}

void* run_encoding_avx2(isa_mt_struct_t* s)
{
    int sz = s->sz;
    int k = s->k;
    int m = s->m;
    u8* g_tbls = s->g_tbls;
    u8**k_buffs = s->k_buffs;
    u8**r_buffs = s->r_buffs;
    pthread_barrier_t* barrier = s->barrier;
    pthread_barrier_wait(barrier);
    //printf("thread ready (avx2)\n");

    while (1)
    {
        //printf("waiting orders\n");
        pthread_barrier_wait(barrier); // this barrier triggers the code execution
        if (*(s->stop) >0) {
            //printf("stop\n");
            return NULL;
        }
        //
        ec_encode_data_avx2(sz, k, m, g_tbls, k_buffs, r_buffs);

        pthread_barrier_wait(barrier);
    }
    return NULL;
}

#ifdef HAVE_AS_KNOWS_AVX512
void* run_encoding_avx512(isa_mt_struct_t* s)
{
    int sz = s->sz;
    int k = s->k;
    int m = s->m;
    u8* g_tbls = s->g_tbls;
    u8**k_buffs = s->k_buffs;
    u8**r_buffs = s->r_buffs;
    pthread_barrier_t* barrier = s->barrier;
    pthread_barrier_wait(barrier);
    //printf("thread ready (avx2)\n");

    while (1)
    {
        //printf("waiting orders\n");
        pthread_barrier_wait(barrier); // this barrier triggers the code execution
        if (*(s->stop) >0) {
            //printf("stop\n");
            return NULL;
        }
        //
        ec_encode_data_avx512(sz, k, m, g_tbls, k_buffs, r_buffs);

        pthread_barrier_wait(barrier);
    }
    return NULL;
}
#endif

void* run_decoding_sse(isa_mt_struct_t* s)
{
    int sz = s->sz;
    int k = s->k;
    int m = s->m;
    size_t start = s->start;
    u8* g_tbls = s->g_tbls;
    u8**k_buffs = s->k_buffs;
    u8**r_buffs = s->r_buffs;
    pthread_barrier_t* barrier = s->barrier;
    pthread_barrier_wait(barrier);
    //printf("thread ready (avx2)\n");
    int i;

    while (1)
    {
        //printf("waiting orders\n");
        pthread_barrier_wait(barrier); // this barrier triggers the code execution
        if (*(s->stop) >0) {
            //printf("stop\n");
            return NULL;
        }

        for (i=0; i < k; i++)
        {
            k_buffs[i] = s->copy_kbuffs[i] + start;
        }
        //
        ec_encode_data_sse(sz, k, m, g_tbls, k_buffs, r_buffs);

        pthread_barrier_wait(barrier);
    }
    return NULL;
}

void* run_decoding_avx(isa_mt_struct_t* s)
{
    int sz = s->sz;
    int k = s->k;
    int m = s->m;
    size_t start = s->start;
    u8* g_tbls = s->g_tbls;
    u8**k_buffs = s->k_buffs;
    u8**r_buffs = s->r_buffs;
    pthread_barrier_t* barrier = s->barrier;
    pthread_barrier_wait(barrier);
    //printf("thread ready (avx2)\n");
    int i;

    while (1)
    {
        //printf("waiting orders\n");
        pthread_barrier_wait(barrier); // this barrier triggers the code execution
        if (*(s->stop) >0) {
            //printf("stop\n");
            return NULL;
        }

        for (i=0; i < k; i++)
        {
            k_buffs[i] = s->copy_kbuffs[i] + start;
        }
        //
        ec_encode_data_avx(sz, k, m, g_tbls, k_buffs, r_buffs);

        pthread_barrier_wait(barrier);
    }
    return NULL;
}

void* run_decoding_avx2(isa_mt_struct_t* s)
{
    int sz = s->sz;
    int k = s->k;
    int m = s->m;
    size_t start = s->start;
    u8* g_tbls = s->g_tbls;
    u8**k_buffs = s->k_buffs;
    u8**r_buffs = s->r_buffs;
    pthread_barrier_t* barrier = s->barrier;
    pthread_barrier_wait(barrier);
    //printf("thread ready (avx2)\n");
    int i;

    while (1)
    {
        //printf("waiting orders\n");
        pthread_barrier_wait(barrier); // this barrier triggers the code execution
        if (*(s->stop) >0) {
            //printf("stop\n");
            return NULL;
        }

        for (i=0; i < k; i++)
        {
            k_buffs[i] = s->copy_kbuffs[i] + start;
        }
        //
        ec_encode_data_avx2(sz, k, m, g_tbls, k_buffs, r_buffs);

        pthread_barrier_wait(barrier);
    }
    return NULL;
}

void* run_decoding_avx512(isa_mt_struct_t* s)
{
    int sz = s->sz;
    int k = s->k;
    int m = s->m;
    size_t start = s->start;
    u8* g_tbls = s->g_tbls;
    u8**k_buffs = s->k_buffs;
    u8**r_buffs = s->r_buffs;
    pthread_barrier_t* barrier = s->barrier;
    pthread_barrier_wait(barrier);
    //printf("thread ready (avx2)\n");
    int i;

    while (1)
    {
        //printf("waiting orders\n");
        pthread_barrier_wait(barrier); // this barrier triggers the code execution
        if (*(s->stop) >0) {
            //printf("stop\n");
            return NULL;
        }

        for (i=0; i < k; i++)
        {
            k_buffs[i] = s->copy_kbuffs[i] + start;
        }
        //
        //ec_encode_data_avx512(sz, k, m, g_tbls, k_buffs, r_buffs);

        pthread_barrier_wait(barrier);
    }
    return NULL;
}


void run_threads(pthread_barrier_t *b)
{
    // wait
    // triggers the coding process in the threads (they are waiting)
    pthread_barrier_wait(b);

    pthread_barrier_wait(b); // synchronize the end of process
    // wait
}


/*
void ec_encode_perf(int m, int k, u8 * a, u8 * g_tbls, u8 ** buffs, struct perf *start)
{
    ec_init_tables(k, m - k, &a[k * k], g_tbls);
    BENCHMARK(start, BENCHMARK_TIME,
              ec_encode_data(TEST_LEN(m), k, m - k, g_tbls, buffs, &buffs[k]));
}

int ec_decode_perf(int m, int k, u8 * a, u8 * g_tbls, u8 ** buffs, u8 * src_in_err,
                   u8 * src_err_list, int nerrs, u8 ** temp_buffs, struct perf *start)
{
    int i, j, r;
    u8 b[MMAX * KMAX], c[MMAX * KMAX], d[MMAX * KMAX];
    u8 *recov[TEST_SOURCES];

    // Construct b by removing error rows
    for (i = 0, r = 0; i < k; i++, r++) {
        while (src_in_err[r])
            r++;
        recov[i] = buffs[r];
        for (j = 0; j < k; j++)
            b[k * i + j] = a[k * r + j];
    }

    if (gf_invert_matrix(b, d, k) < 0)
        return BAD_MATRIX;

    for (i = 0; i < nerrs; i++)
        for (j = 0; j < k; j++)
            c[k * i + j] = d[k * src_err_list[i] + j];

    // Recover data
    ec_init_tables(k, nerrs, c, g_tbls);
    BENCHMARK(start, BENCHMARK_TIME,
              ec_encode_data(TEST_LEN(m), k, nerrs, g_tbls, recov, temp_buffs));

    return 0;
}
*/


int main(int argc, char *argv[])
{
    int i, j, rtest, nerrs, r, check;
    void *buf;
    u8 *temp_buffs[TEST_SOURCES], *buffs[TEST_SOURCES];
    u8 a[MMAX * KMAX], b[MMAX * KMAX], c[MMAX * KMAX], d[MMAX * KMAX];
    u8 g_tbls[KMAX * TEST_SOURCES * 32], src_in_err[TEST_SOURCES];
    u8 src_err_list[TEST_SOURCES], *recov[TEST_SOURCES];
    struct perf start;

    // Pick test parameters
    m = 14;
    k = 10;
    nerrs = 100;
    t = 0;
    l=1024;
    const u8 err_list[] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99 };
    parse_args(argc, argv);
    m = k+rr;
    if (k< rr)
        nerrs = k;
    else
        nerrs = rr;
    print_params();
    printf("m:%d\n",m);

    printf("ec_perf_runtime_example: %dx%d %d\n", m, sz, nerrs);

    if (m > MMAX || k > KMAX || nerrs > (m - k)) {
        printf(" Input test parameter error\n");
        return -1;
    }

    memcpy(src_err_list, err_list, nerrs);
    memset(src_in_err, 0, TEST_SOURCES);
    for (i = 0; i < nerrs; i++)
        src_in_err[src_err_list[i]] = 1;

    // Allocate the arrays
    for (i = 0; i < m; i++) {
        if (posix_memalign(&buf, 64, sz)) {
            printf("alloc error: Fail\n");
            return -1;
        }
        buffs[i] = buf;
    }

    for (i = 0; i < (m - k); i++) {
        if (posix_memalign(&buf, 64, sz)) {
            printf("alloc error: Fail\n");
            return -1;
        }
        temp_buffs[i] = buf;
    }

    // Make random data
    for (i = 0; i < k; i++)
        for (j = 0; j < sz; j++)
            buffs[i][j] = rand();

    gf_gen_rs_matrix(a, m, k);



 /*
    // Start encode test
    ec_encode_perf(m, k, a, g_tbls, buffs, &start);
    printf("erasure_code_encode" TEST_TYPE_STR ": ");
    perf_print(start, (long long)(sz) * (k + nerrs) * rtest);

    // Start decode test
    check = ec_decode_perf(m, k, a, g_tbls, buffs, src_in_err, src_err_list, nerrs,
                           temp_buffs, &start);

    if (check == BAD_MATRIX) {
        printf("BAD MATRIX\n");
        return check;
    }
*/

    pthread_barrier_t enc_barrier, dec_barrier;
    pthread_t *threads;
    int stop_threads = 0;
    // run threads, instruction set was defined => same code for sse,avx,avx2
    if (t>0)
    {
        pthread_barrier_init(&(enc_barrier), NULL, t+1);

        threads = malloc(sizeof(pthread_t)*t);
        // create the encoding threads, the barrier
        for (i = 0; i < t; i++)
        {
            isa_mt_struct_t * mt_params = create_mt_encoding_struct(i,t, sz, k, m - k, g_tbls, buffs, &buffs[k], &enc_barrier,&stop_threads);

            switch(instr)
            {
                case INTEL_SSE:
                    pthread_create(&(threads[i]), NULL,  (void *(*)(void*))run_encoding_sse, (void*)mt_params);
                    break;
                case INTEL_AVX:
                    pthread_create(&(threads[i]), NULL,  (void *(*)(void*))run_encoding_avx, (void*)mt_params);
                    break;
                case INTEL_AVX2:
                    pthread_create(&(threads[i]), NULL,  (void *(*)(void*))run_encoding_avx2, (void*)mt_params);
                    break;
#ifdef HAVE_AS_KNOWS_AVX512
                case INTEL_AVX512:
                    pthread_create(&(threads[i]), NULL,  (void *(*)(void*))run_encoding_avx512, (void*)mt_params);
                    break;
#endif
            }

            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i, &cpuset);


            int s = pthread_setaffinity_np(threads[i], sizeof(cpu_set_t), &cpuset);

            pthread_detach(threads[i]);
        }

        ec_init_tables(k, m - k, &a[k * k], g_tbls);
        pthread_barrier_wait(&enc_barrier);

        perf_start(&start);
        for (rtest = 0; rtest < l; rtest++) {
            // Make parity vects

            run_threads(&enc_barrier);
        }
        perf_stop(&start);
        printf("ec_perf_encode: ");
        perf_print(start, (long long)(sz) * (m) * rtest);


        // signal the threads to stop
        stop_threads = 1;

        //trigger the barrier
        pthread_barrier_wait(&enc_barrier);
        usleep(100000);
        stop_threads = 0;



        pthread_barrier_init(&(dec_barrier), NULL, t+1);
        // create the decoding threads
        for (i = 0; i < t; i++)
        {
            isa_mt_struct_t * mt_params = create_mt_decoding_struct(i,t, sz, k, nerrs, g_tbls, recov, temp_buffs, &dec_barrier, &stop_threads);

            switch(instr)
            {
                case INTEL_SSE:
                    pthread_create(&(threads[i]), NULL,  (void *(*)(void*))run_decoding_sse, (void*)mt_params);
                    break;
                case INTEL_AVX:
                    pthread_create(&(threads[i]), NULL,  (void *(*)(void*))run_decoding_avx, (void*)mt_params);
                    break;
                case INTEL_AVX2:
                    pthread_create(&(threads[i]), NULL,  (void *(*)(void*))run_decoding_avx2, (void*)mt_params);
                    break;
#ifdef HAVE_AS_KNOWS_AVX512
                case INTEL_AVX512:
                    pthread_create(&(threads[i]), NULL,  (void *(*)(void*))run_decoding_avx512, (void*)mt_params);
                    break;
#endif
            }

            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i, &cpuset);


            int s = pthread_setaffinity_np(threads[i], sizeof(cpu_set_t), &cpuset);

            pthread_detach(threads[i]);
        }


        pthread_barrier_wait(&dec_barrier);
        // Start decode test
        perf_start(&start);
        for (rtest = 0; rtest < l; rtest++) {
            // Construct b by removing error rows
            for (i = 0, r = 0; i < k; i++, r++) {
                while (src_in_err[r])
                    r++;
                recov[i] = buffs[r];
                for (j = 0; j < k; j++)
                    b[k * i + j] = a[k * r + j];
            }

            if (gf_invert_matrix(b, d, k) < 0) {
                printf("BAD MATRIX\n");
                return -1;
            }

            for (i = 0; i < nerrs; i++)
                for (j = 0; j < k; j++)
                    c[k * i + j] = d[k * src_err_list[i] + j];

            // Recover data
            ec_init_tables(k, nerrs, c, g_tbls);
            run_threads(&dec_barrier);
        }
        perf_stop(&start);

    }else{
        switch(instr)
        {
            case INTEL_SSE:
                // Start encode test
                perf_start(&start);
                for (rtest = 0; rtest < l; rtest++) {
                    // Make parity vects
                    ec_init_tables(k, m - k, &a[k * k], g_tbls);
                    ec_encode_data_sse(sz, k, m - k, g_tbls, buffs, &buffs[k]);
                }
                perf_stop(&start);
                printf("ec_perf_encode: ");
                perf_print(start, (long long)(sz) * (m) * rtest);
                // Start decode test
                perf_start(&start);
                for (rtest = 0; rtest < l; rtest++) {
                    // Construct b by removing error rows
                    for (i = 0, r = 0; i < k; i++, r++) {
                        while (src_in_err[r])
                            r++;
                        recov[i] = buffs[r];
                        for (j = 0; j < k; j++)
                            b[k * i + j] = a[k * r + j];
                    }

                    if (gf_invert_matrix(b, d, k) < 0) {
                        printf("BAD MATRIX\n");
                        return -1;
                    }

                    for (i = 0; i < nerrs; i++)
                        for (j = 0; j < k; j++)
                            c[k * i + j] = d[k * src_err_list[i] + j];

                    // Recover data
                    ec_init_tables(k, nerrs, c, g_tbls);
                    ec_encode_data_sse(sz, k, nerrs, g_tbls, recov, temp_buffs);
                }
                perf_stop(&start);
                break;
            case INTEL_AVX:
                // Start encode test
                perf_start(&start);
                for (rtest = 0; rtest < l; rtest++) {
                    // Make parity vects
                    ec_init_tables(k, m - k, &a[k * k], g_tbls);
                    ec_encode_data_avx(sz, k, m - k, g_tbls, buffs, &buffs[k]);
                }
                perf_stop(&start);
                printf("ec_perf_encode: ");
                perf_print(start, (long long)(sz) * (m) * rtest);

                // Start decode test
                perf_start(&start);
                for (rtest = 0; rtest < l; rtest++) {
                    // Construct b by removing error rows
                    for (i = 0, r = 0; i < k; i++, r++) {
                        while (src_in_err[r])
                            r++;
                        recov[i] = buffs[r];
                        for (j = 0; j < k; j++)
                            b[k * i + j] = a[k * r + j];
                    }

                    if (gf_invert_matrix(b, d, k) < 0) {
                        printf("BAD MATRIX\n");
                        return -1;
                    }

                    for (i = 0; i < nerrs; i++)
                        for (j = 0; j < k; j++)
                            c[k * i + j] = d[k * src_err_list[i] + j];

                    // Recover data
                    ec_init_tables(k, nerrs, c, g_tbls);
                    ec_encode_data_avx(sz, k, nerrs, g_tbls, recov, temp_buffs);
                }
                perf_stop(&start);
                break;
            case INTEL_AVX2:
                // Start encode test
                perf_start(&start);
                for (rtest = 0; rtest < l; rtest++) {
                    // Make parity vects
                    ec_init_tables(k, m - k, &a[k * k], g_tbls);
                    ec_encode_data_avx2(sz, k, m - k, g_tbls, buffs, &buffs[k]);
                }
                perf_stop(&start);
                printf("ec_perf_encode: ");
                perf_print(start, (long long)(sz) * (m) * rtest);

                // Start decode test
                perf_start(&start);
                for (rtest = 0; rtest < l; rtest++) {
                    // Construct b by removing error rows
                    for (i = 0, r = 0; i < k; i++, r++) {
                        while (src_in_err[r])
                            r++;
                        recov[i] = buffs[r];
                        for (j = 0; j < k; j++)
                            b[k * i + j] = a[k * r + j];
                    }

                    if (gf_invert_matrix(b, d, k) < 0) {
                        printf("BAD MATRIX\n");
                        return -1;
                    }

                    for (i = 0; i < nerrs; i++)
                        for (j = 0; j < k; j++)
                            c[k * i + j] = d[k * src_err_list[i] + j];

                    // Recover data
                    ec_init_tables(k, nerrs, c, g_tbls);
                    ec_encode_data_avx2(sz, k, nerrs, g_tbls, recov, temp_buffs);
                }
                perf_stop(&start);
                break;
#ifdef HAVE_AS_KNOWS_AVX512
             case INTEL_AVX512:
                 // Start encode test
                perf_start(&start);
                for (rtest = 0; rtest < l; rtest++) {
                    // Make parity vects
                    ec_init_tables(k, m - k, &a[k * k], g_tbls);
                    ec_encode_data_avx512(sz, k, m - k, g_tbls, buffs, &buffs[k]);
                }
                perf_stop(&start);
                printf("ec_perf_encode: ");
                perf_print(start, (long long)(sz) * (m) * rtest);

                // Start decode test
                perf_start(&start);
                for (rtest = 0; rtest < l; rtest++) {
                    // Construct b by removing error rows
                    for (i = 0, r = 0; i < k; i++, r++) {
                        while (src_in_err[r])
                            r++;
                        recov[i] = buffs[r];
                        for (j = 0; j < k; j++)
                            b[k * i + j] = a[k * r + j];
                    }

                    if (gf_invert_matrix(b, d, k) < 0) {
                        printf("BAD MATRIX\n");
                        return -1;
                    }

                    for (i = 0; i < nerrs; i++)
                        for (j = 0; j < k; j++)
                            c[k * i + j] = d[k * src_err_list[i] + j];

                    // Recover data
                    ec_init_tables(k, nerrs, c, g_tbls);
                    ec_encode_data_avx512(sz, k, nerrs, g_tbls, recov, temp_buffs);
                }
                perf_stop(&start);
                break;
#endif
        }
    }
    for (i = 0; i < nerrs; i++) {
        if (0 != memcmp(temp_buffs[i], buffs[src_err_list[i]], sz)) {
            printf("Fail error recovery (%d, %d, %d) - ", m, k, nerrs);
            return -1;
        }
    }

    printf("ec_perf_decode: ");
    perf_print(start, (long long)(sz) * (k + nerrs) * rtest);

    printf("done all: Pass\n");
    return 0;
}
