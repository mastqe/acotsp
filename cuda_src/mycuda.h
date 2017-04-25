/* Declarations needed for GPU code */

#include <curand.h>
#include <curand_kernel.h>

typedef struct d_problem {
  long int      n;                      /* number of cities */
  long int      n_near;                 /* number of nearest neighbors */
  long int      **distance;	        /* distance matrix: distance[i][j] gives distance
					   between city i und j */
  long int      **nn_list;              /* nearest neighbor list; contains for each node i a
                                           sorted list of n_near nearest neighbors */
} d_problem_t;

extern d_problem_t* instance_to_d;
extern d_problem_t  instance_for_d;

extern ant_struct *ant_to_d;      /* this (array of) struct will hold the colony on the GPU */

extern double **total_to_d;

__device__ 
void ant_empty_memory( ant_struct *a );

__device__
void place_ant( ant_struct *a , long int phase );

__device__
void choose_best_next( ant_struct *a, long int phase );

__device__
void neighbour_choose_best_next( ant_struct *a, long int phase );

__device__
void neighbour_choose_and_move_to_next( ant_struct *a , long int phase );

__global__
void construct_solutions( ant_struct* ant_to_d, d_problem_t* instance_to_d, double** total_to_d );

__global__ 
void rand_init();

void init_device();

/******************************************************************************/
/*** helper code from http://cs.txstate.edu/~burtscher/research/TSP_GPU/ ******/
/******************************************************************************/

void CudaTest(char *msg);

#define mallocOnGPU(addr, size) if (cudaSuccess != cudaMalloc((void **)&addr, size)) fprintf(stderr, "could not allocate GPU memory\n");  CudaTest("couldn't allocate GPU memory");
#define copyToGPU(to, from, size) if (cudaSuccess != cudaMemcpy(to, from, size, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of data to device failed\n");  CudaTest("data copy to device failed");
#define copyFromGPU(to, from, size) if (cudaSuccess != cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of data from device failed\n");  CudaTest("data copy from device failed");
#define copyFromGPUSymbol(to, from, size) if (cudaSuccess != cudaMemcpyFromSymbol(to, from, size)) fprintf(stderr, "copying of symbol from device failed\n");  CudaTest("symbol copy from device failed");
#define copyToGPUSymbol(to, from, size) if (cudaSuccess != cudaMemcpyToSymbol(to, from, size)) fprintf(stderr, "copying of symbol to device failed\n");  CudaTest("symbol copy to device failed");

/******************************************************************************/
