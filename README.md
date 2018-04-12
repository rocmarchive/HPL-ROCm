-- High Performance Computing Linpack Benchmark for HPL
    hpl-ROCm - 0.001 - 2017

    David Martin (cuda@avidday.net)
    (C) Copyright 2010-2011 All Rights Reserved
    For HIP Port 
    (C). AMD 2018 All Rights Reserved  

See the accompanying COPYING file for full details of the
license and copyright information of the code contained in
this distribution.

This distribution contains a simple acceleration scheme for
the standard HPL-2.0 benchmark with a double precision capable
AMD GPU and the rocBLAS library.

The code has been known to build on Ubuntu 16.04 LTS or later and Redhat 7.4
and derivatives, using mpich2 and GotoBLAS, with ROCm. 1.7.1 or later.

The supplied Make.CUDA file relies on a number of environment variables
being set to correctly locate host BLAS and MPI, and rocBLAS libraries
and include files. Example values for the abovementioned mpich2 and gotoBLAS
combinations might be:

MPICH2_INSTALL_PATH=/opt/mpich2-1.3
MPICH2_INCLUDES=-I/opt/mpich2-1.3/include
MPICH2_LIBRARIES=-L/opt/mpich2-1.3/lib -lmpich -lmpl
BLAS_INCLUDES=-I/opt/goto/GotoBLAS2/include
BLAS_LIBRARIES=-L/opt/goto/GotoBLAS2/lib -lgoto2 -lgfortran -lpthread
CUBLAS_INCLUDES=-I/opt/cuda-3.2/include
CUBLAS_LIBRARIES=-L/opt/cuda-3.2/lib64 -lcublas -lcudart -lcuda

With these set, it should be possible to build by issuing

$ make arch=CUDA

which will drop the final xhpl executable in the bin/CUDA directory.
Correct runtime path and link load settings are the responsibility
of the user.

ROCm Support:

The ROCm port requires the ROCm software stack and the libopenblas-dev
blas implementation. The libopenblas-dev package can be installed on
Ubuntu with this command:

$ sudo apt-get install libopenblas-dev

To build with ROCm support instead of Cuda support use the ROCm arch:

$ source envsetup.sh
$ make arch=ROCm

NOTE:

This version of the code only acclerates one section
of the factorization for a single GPU. The scheduling routine
gpuUpdatePlanCreate() in auxil/HPL_gpusupport.c contains two
tuning constants tune0 and tune1, which control the split
of work between the GPU and host CPU(s). These must be tuned
for optimal performance with a given GPU and host CPU/BLAS
combination. How this is done is left as an exercise for the
reader.

There are a number of further optimizations which can be applied
to this code - it should be regarded as a starting point rather
than a definitive version of the benchmark.

DISCLAIMER:

 THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT 
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
 OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
 SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT 
 LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY 
 THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

