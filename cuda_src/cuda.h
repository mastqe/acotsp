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
extern __device__ d_problem_t* d_instance;

extern __device__ ant_struct* d_ant;
extern ant_struct *ant_to_d;      /* this (array of) struct will hold the colony on the GPU */

extern double **total_to_d;
extern __device__ double **d_total;     /* combination of pheromone times heuristic information */

extern __device__ curandState_t state;

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
