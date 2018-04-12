/*
 *  -- High Performance Computing Linpack Benchmark for CUDA
 *    hpl-cuda - 0.001 - 2010
 *
 *    David Martin (cuda@avidday.net)
 *    (C) Copyright 2010-2011 All Rights Reserved
 *
 *    portions of this code written by
 *
 *    Antoine P. Petitet
 *    University of Tennessee, Knoxville
 *    Innovative Computing Laboratory
 *    (C) Copyright 2000-2008 All Rights Reserved
 *
 *    portions of this code written by Vasily Volkov.
 *    Copyright (c) 2009, The Regents of the University of California.
 *    All rights reserved.
 *
 * -- Copyright notice and Licensing terms:
 *
 * Redistribution  and  use in  source and binary forms, with or without
 * modification, are  permitted provided  that the following  conditions
 * are met:
 *
 * 1. Redistributions  of  source  code  must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce  the above copyright
 * notice, this list of conditions,  and the following disclaimer in the 
 * documentation and/or other materials provided with the distribution.
 *
 * 3. All  advertising  materials  mentioning  features  or  use of this
 * software must display the following acknowledgements:
 * This  product  includes  software  developed  at  the  University  of  
 * Tennessee, Knoxville, Innovative Computing Laboratory.
 * This product  includes software  developed at the Frankfurt Institute
 * for Advanced Studies.
 */
#include <cuda.h>
#ifdef HPL_GPU_ROCM
#include <hip/hip_runtime.h>
#include <hipblas.h>
#else
#include <cuda_runtime.h>
#include <cublas.h>
#endif

#include "hpl.h"
#include "hpl_gpusupport.h"

const unsigned int MB = 1<<20;
static unsigned int reserved, allocated;
static char *pool = NULL;
static int initialised = gpuWarn;

/*
 *  Reset the partitioning of the GPU memory
 */
void gpu_malloc_reset( )
{
    allocated = 0;
}

void gpu_release( )
{
    if( pool != NULL )
    {
#ifdef HPL_GPU_ROCM
        gpuQ( hipFree( pool ) );
#else
        gpuQ( cudaFree( pool ) );
#endif
        pool = NULL;
    }
    reserved = allocated = 0;
}

/*
 *  Allocate all resources that we might ever need
 */
int gpu_init( char warmup )
{
    (void)warmup;
    initialised = gpuWarn;

    gpu_release( );
    
#ifdef HPL_GPU_ROCM
    if( hipblasCreate(&blasHandle) != HIPBLAS_STATUS_SUCCESS )
#else
    if( cublasInit() != CUBLAS_STATUS_SUCCESS )
#endif
    {
        initialised = gpuFail;
        return initialised;
    }
    
#ifdef HPL_CUDA_DIAGNOSTICS
    int idevice;
#ifdef HPL_GPU_ROCM
    struct hipDeviceProp_t prop;
    gpuQ( hipGetDevice( &idevice ) );
    gpuQ( hipGetDeviceProperties( &prop, idevice ) );
#else
    struct cudaDeviceProp prop;
    gpuQ( cudaGetDevice( &idevice ) );
    gpuQ( cudaGetDeviceProperties( &prop, idevice ) );
#endif
    HPL_fprintf(stderr, "HPL_gpusupport.c: using device %s, %.0f MHz clock, %.0f MB memory.\n", prop.name, prop.clockRate/1000.f, prop.totalGlobalMem/1048576.f);
#endif

    /*
     *  pre-allocate as much GPU memory as possible
     */
    unsigned int total;
#ifdef HPL_GPU_ROCM
    size_t res, tot;
    gpuQ( hipMemGetInfo( &res, &tot ) );
    reserved = (unsigned int) res;
    total = (unsigned int) tot;
    while( hipMalloc( (void**)&pool, reserved ) != hipSuccess )
#else
    gpuQ( cuMemGetInfo( &reserved, &total ) );
    while( cudaMalloc( (void**)&pool, reserved ) != cudaSuccess )
#endif
    {
        reserved -= MB;
        if( reserved < MB )
        {
#ifdef HPL_GPU_ROCM
            gpuQ( hipblasDestroy(blasHandle) );
#else
            gpuQ( cublasShutdown() );
#endif
            initialised = gpuFail;
            return initialised;
        }
    }
    
    /* This takes up overheads that would otherwise reduce the performance
     * of "real" calls. Using double precision should ensure we get a warning 
     * or error if trying to use a < 1.3 compute capable device.
     */
    if ( warmup ) {
        int m = 128, n = 1;
        struct gpuArray junk;
        gpuQ( gpu_malloc2D(m,n,&junk) );
#ifdef HPL_GPU_ROCM
        const double two = 2.0;
        hipblasDscal( blasHandle, m, &two, junk.ptr, 1 );
        gpuQ( hipDeviceSynchronize() );
#else
        cublasDscal( m, 2., junk.ptr, 1 );
        gpuQ( cublasGetError() );
        gpuQ( cudaThreadSynchronize() );
#endif
        gpu_malloc_reset();
    }

    /*
     *  reset the error states
     */
#ifdef HPL_GPU_ROCM
    hipGetLastError();
#else
    cudaGetLastError();
#endif

    initialised = gpuPass;
    return initialised;
}

int gpu_ready(unsigned int *memory )
{
    char warmup = 0;

    *memory = reserved;

    if (initialised == gpuWarn)
        return gpu_init(warmup);
    else
        return initialised;
}
        
/*
 *  Get a piece of the allocated GPU memory
 */
int gpu_malloc2D( int m, int n, struct gpuArray *p )
{

    if( m <= 0 || n <= 0 ) {
        p->ptr = NULL;
        p->lda = 0;
        p->size = 0;
        return gpuFail;
    }
    
    /*
     *  pad to align columns and avoid bank contention
     */
    unsigned int m_padded = (m+63)&~63|64;
    
    /*
     *  pad to avoid page failures in custom BLAS3 kernels
     */
    unsigned int n_padded = (n+31)&~31;
    
    /*
     *  allocate memory
     */
    unsigned int size = sizeof(double) * m_padded * n_padded;
    if( allocated + size <= reserved )
    {
        p->ptr = (double*)(pool + allocated);
        p->lda = m_padded;
        p->size = size;

        allocated = (allocated + size + 63) & ~63;

        return gpuPass;

    } else {
        p->ptr = NULL;
        p->lda = 0;
        p->size = 0;
        return gpuFail;
    } 
}

void gpu_upload( const int m, const int n, struct gpuArray *dst, const double *src, const int srclda)
{       
    if( m > 0 && n > 0 )
#ifdef HPL_GPU_ROCM
        gpuQ( hipMemcpy2D( dst->ptr, dst->lda*sizeof(double),
                            src, srclda*sizeof(double),
                            m*sizeof(double), n, hipMemcpyHostToDevice ) );
#else
        gpuQ( cudaMemcpy2D( dst->ptr, dst->lda*sizeof(double), 
                            src, srclda*sizeof(double),
                            m*sizeof(double), n, cudaMemcpyHostToDevice ) );
#endif
}       

void gpu_download( const int m, const int n, double *dst, const int dstlda, struct gpuArray *src )
{       
    if( m > 0 && n > 0 )
#ifdef HPL_GPU_ROCM
        gpuQ( hipMemcpy2D( dst, dstlda*sizeof(double),
                            src->ptr, src->lda*sizeof(double),
                            m*sizeof(double), n, hipMemcpyDeviceToHost ) );
#else
        gpuQ( cudaMemcpy2D( dst, dstlda*sizeof(double), 
                            src->ptr, src->lda*sizeof(double), 
                            m*sizeof(double), n, cudaMemcpyDeviceToHost ) );
#endif
}       

void gpu_copy( const int m, const int n, struct gpuArray *dst, struct gpuArray *src )
{       
    if( m > 0 && n > 0 )
#ifdef HPL_GPU_ROCM
        gpuQ( hipMemcpy2D( dst->ptr, dst->lda*sizeof(double),
                            src->ptr, src->lda*sizeof(double),
                            m*sizeof(double), n, hipMemcpyDeviceToDevice ) );
#else
        gpuQ( cudaMemcpy2D( dst->ptr, dst->lda*sizeof(double), 
                            src->ptr, src->lda*sizeof(double), 
                            m*sizeof(double), n, cudaMemcpyDeviceToDevice ) );
#endif
}       

struct gpuUpdatePlan * gpuUpdatePlanCreate(int mp, int nn, int jb)
{
    (void)nn; (void)jb;

    struct gpuUpdatePlan * plan = (struct gpuUpdatePlan *)calloc((size_t)1, sizeof(struct gpuUpdatePlan));

    if ( (gpu_ready( &(plan->availMemory) ) == gpuPass) && (mp > 0) ) {
        plan->strategy = gpuBoth;

        gpuQ( gpu_malloc2D( mp, jb, &(plan->gL2) ) );
        gpuQ( gpu_malloc2D( jb, nn, &(plan->gA) ) );
        gpuQ( gpu_malloc2D( jb, jb, &(plan->gL1) ) );

        int memused = plan->gL2.size + plan->gA.size + plan->gL1.size;

        plan->nntot = nn;
        plan->nnmax = (plan->availMemory - memused) / (sizeof(double) * (plan->gL2.lda + plan->gA.lda + plan->gL1.lda));

        gpuQ( gpu_malloc2D( mp, plan->nnmax, &(plan->gA2) ) );
            
        const int tune0 = 6, tune1 = 7;
        int nhat = (32*tune0)*(plan->nnmax/(32*tune1));
        plan->nDgemmStages = Mmax( plan->nntot / nhat, 1 );
        int nmod = Mmax( plan->nntot - (plan->nDgemmStages * nhat), 0 ); 
        int nmod1 = nmod / plan->nDgemmStages;
        if ( nmod1  > (nhat/tune1) ) {
            plan->nDgemmStages++;
            nmod = Mmax( plan->nntot - (plan->nDgemmStages * nhat), 0 ); 
            nmod1 = nmod / plan->nDgemmStages;
        }
        int nmod0 = nmod - nmod1*plan->nDgemmStages;

        int iter, n0;
        plan->dgemmStages = (struct gpuDgemmStage *)calloc( (size_t)plan->nDgemmStages, sizeof(struct gpuDgemmStage) );
        for( n0=0,iter=0; iter<plan->nDgemmStages; iter++ ) {
            int nhati = Mmin((plan->nntot-n0), nhat);
            plan->dgemmStages[iter].N1 = tune0*(nhati/tune1); 
            plan->dgemmStages[iter].N2 = nhati - plan->dgemmStages[iter].N1 + nmod1 + nmod0;
            n0 += nhati + nmod1 + nmod0;
            nmod0 = 0;
        }

    } else {
        plan->strategy = gpuNone;
    }

    return plan;
}

void gpuUpdatePlanDestroy(struct gpuUpdatePlan * plan)
{ 
    if (plan != NULL) {
        if (plan->dgemmStages != NULL)
            free(plan->dgemmStages);
        free(plan); 
    }

    gpu_malloc_reset( );
}

void gpuDgemmPlanStage(struct gpuUpdatePlan * plan, const int stage, int *N1, int *N2)
{
    if (stage < plan->nDgemmStages) {
        *N1 = plan->dgemmStages[stage].N1;
        *N2 = plan->dgemmStages[stage].N2;
    } else {
        *N1 = 0; *N2 = 0;
    }
}

/* vim:ts=4:sw=4:expandtab:number
 */
