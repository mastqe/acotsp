
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <limits.h>
#include <time.h>

#include "InOut.h"
#include "TSP.h"
#include "ants.h"
#include "ls.h"
#include "utilities.h"
#include "timer.h"
#include "mycuda.h"
#include "acotsp.h"


d_problem_t  instance_for_d;
d_problem_t* instance_to_d;
__device__ d_problem_t* d_instance;

__device__ curandState_t state;

ant_struct *ant_to_d;
__device__ ant_struct* d_ant;

double **total_to_d;
__device__ double **d_total;


/******************************************************************************/
/*** helper code from http://cs.txstate.edu/~burtscher/research/TSP_GPU/ ******/
/******************************************************************************/

void CudaTest(char *msg)
{
    cudaError_t e;

    cudaThreadSynchronize();
    if (cudaSuccess != (e = cudaGetLastError())) {
        fprintf(stderr, "%s: %d\n", msg, e); 
        fprintf(stderr, "%s\n", cudaGetErrorString(e));
        exit(-1);
    }
}

/******************************************************************************/

__device__ 
void ant_empty_memory( ant_struct *a )
/*
      FUNCTION:       empty the ants's memory regarding visited cities
      INPUT:          ant identifier
      OUTPUT:         none
      (SIDE)EFFECTS:  vector of visited cities is reinitialized to FALSE
*/
{
    long int   i;

    for( i = 0 ; i < d_instance->n ; i++ ) {
        a->visited[i]=FALSE;
    }
}


__device__
void place_ant( ant_struct *a , long int step )
/*
      FUNCTION:      place an ant on a randomly chosen initial city
      INPUT:         pointer to ant and the number of construction steps
      OUTPUT:        none
      (SIDE)EFFECT:  ant is put on the chosen city
*/
{
    long int     rnd;

    /* random number between 0 .. n-1 */
    rnd = (long int) (curand_uniform(&state) * (double) d_instance->n);

    a->tour[step] = rnd;
    a->visited[rnd] = TRUE;
}


__device__
void choose_best_next( ant_struct *a, long int phase )
/*
      FUNCTION:      chooses for an ant as the next city the one with
                     maximal value of heuristic information times pheromone
      INPUT:         pointer to ant and the construction step
      OUTPUT:        none
      (SIDE)EFFECT:  ant moves to the chosen city
*/
{
    long int city, current_city, next_city;
    double   value_best;

    next_city = d_instance->n;
    DEBUG( assert ( phase > 0 && phase < n ); );
    current_city = a->tour[phase-1];
    value_best = -1.;             /* values in total matrix are always >= 0.0 */

    for ( city = 0 ; city < d_instance->n ; city++ ) {
        if ( a->visited[city] )
            ; /* city already visited, do nothing */
        else {
            if ( d_total[current_city][city] > value_best ) {
                next_city = city;
                value_best = d_total[current_city][city];
            }
        }
    }

    DEBUG( assert ( 0 <= next_city && next_city < n); );
    DEBUG( assert ( value_best > 0.0 ); );
    DEBUG( assert ( a->visited[next_city] == FALSE ); );

    a->tour[phase] = next_city;
    a->visited[next_city] = TRUE;
}


__device__
void neighbour_choose_best_next( ant_struct *a, long int phase )
/*
      FUNCTION:      chooses for an ant as the next city the one with
                     maximal value of heuristic information times pheromone
      INPUT:         pointer to ant and the construction step "phase"
      OUTPUT:        none
      (SIDE)EFFECT:  ant moves to the chosen city
*/
{
    long int i, current_city, next_city, help_city;
    double   value_best, help;

    next_city = d_instance->n;
    DEBUG( assert ( phase > 0 && phase < n ); );
    current_city = a->tour[phase-1];
    DEBUG ( assert ( 0 <= current_city && current_city < n ); );
    value_best = -1.;             /* values in total matix are always >= 0.0 */

    for ( i = 0 ; i < d_instance->n_near ; i++ ) {
        help_city = d_instance->nn_list[current_city][i];
        if ( a->visited[help_city] )
            ;   /* city already visited, do nothing */
        else {
            help = d_total[current_city][help_city];
            if ( help > value_best ) {
                value_best = help;
                next_city = help_city;
            }
        }
    }

    if ( next_city == d_instance->n )
        /* all cities in nearest neighbor list were already visited */
        choose_best_next( a, phase );
    else {
        DEBUG( assert ( 0 <= next_city && next_city < n); );
        DEBUG( assert ( value_best > 0.0 ); );
        DEBUG( assert ( a->visited[next_city] == FALSE ); );

        a->tour[phase] = next_city;
        a->visited[next_city] = TRUE;
    }
}

__device__
void neighbour_choose_and_move_to_next( ant_struct *a , long int phase )
/*
      FUNCTION:      Choose for an ant probabilistically a next city among all
                     unvisited cities in the current city's candidate list.
             If this is not possible, choose the closest next
      INPUT:         pointer to ant the construction step "phase"
      OUTPUT:        none
      (SIDE)EFFECT:  ant moves to the chosen city
*/
{
    long int i, help;
    long int current_city;
    double   rnd, partial_sum = 0., sum_prob = 0.0;
    double   prob_of_selection[NN_ANTS + 1]; /* stores the selection probabilities
                                           of the nearest neighbor cities */
    
    /* Ensures that we do not run over the last element in the random wheel. */
    prob_of_selection[d_instance->n_near] = HUGE_VAL;

    double   *prob_ptr;

    /** NEVER USED in our default case - ignore for simplicity
    if ( (q_0 > 0.0) && (ran01( this_seed ) < q_0)  ) {
        // with a probability q_0 make the best possible choice
        //   according to pheromone trails and heuristic information
        // we first check whether q_0 > 0.0, to avoid the very common case
        //  of q_0 = 0.0 to have to compute a random number, which is
        //  expensive computationally 
        neighbour_choose_best_next(a, phase);
        return;
    }
    */

    prob_ptr = prob_of_selection;

    current_city = a->tour[phase-1]; /* current_city city of ant k */

    DEBUG( assert ( current_city >= 0 && current_city < n ); );

    for ( i = 0 ; i < d_instance->n_near ; i++ ) {
        if ( a->visited[d_instance->nn_list[current_city][i]] )
            prob_ptr[i] = 0.0;   /* city already visited */
        else {
            DEBUG( assert ( instance.nn_list[current_city][i] >= 0 && instance.nn_list[current_city][i] < n ); );

            prob_ptr[i] = d_total[current_city][d_instance->nn_list[current_city][i]];
            sum_prob += prob_ptr[i];
        }
    }

    if (sum_prob <= 0.0) {
        /* All cities from the candidate set are tabu */
        choose_best_next( a, phase );
    }
    else {
        /* at least one neighbor is eligible, chose one according to the
           selection probabilities */

        rnd = curand_uniform(&state);

        rnd *= sum_prob;
        i = 0;
        partial_sum = prob_ptr[i];

        /* This loop always stops because prob_ptr[nn_ants] == HUGE_VAL  */
        while (partial_sum <= rnd) {
            i++;
            partial_sum += prob_ptr[i];
        }

        /* This may very rarely happen because of rounding if rnd is
           close to 1.  */
        if (i == d_instance->n_near) {
            neighbour_choose_best_next(a, phase);
            return;
        }

        DEBUG( assert ( 0 <= i && i < nn_ants); );
        DEBUG( assert ( prob_ptr[i] >= 0.0); );

        help = d_instance->nn_list[current_city][i];

        DEBUG( assert ( help >= 0 && help < n ); );
        DEBUG( assert ( a->visited[help] == FALSE ); );

        a->tour[phase] = help; /* instance.nn_list[current_city][i]; */
        a->visited[help] = TRUE;
    }
}

__device__
long int d_compute_tour_length( long int *t )
/*
      FUNCTION: compute the tour length of tour t
      INPUT:    pointer to tour t
      OUTPUT:   tour length of tour t
*/
{
    int      i;
    long int tour_length = 0;

    for ( i = 0 ; i < d_instance->n ; i++ ) {
        tour_length += d_instance->distance[t[i]][t[i+1]];
    }
    return tour_length;
}

__global__
void construct_solutions( ant_struct* ant_to_d, d_problem_t* instance_to_d, double** total_to_d )
/*
      FUNCTION:       CUDA Kernel to manage the solution construction phase
      INPUT:          none
      OUTPUT:         none
      (SIDE)EFFECTS:  when finished, all ants of the colony have constructed a solution
*/
{
    // make parameters global
    d_ant = ant_to_d;
    d_instance = instance_to_d;
    d_total = total_to_d;

    long int step;    /* counter of the number of construction steps */

    TRACE ( printf("construct solutions for all ants\n"); );

    /* Mark all cities as unvisited */
    ant_empty_memory( &d_ant[threadIdx.x] );

    /* Place the ants on random initial city */
    place_ant( &d_ant[threadIdx.x], 0 ); 

    for ( step = 1; step < d_instance->n; step++ ) {
        neighbour_choose_and_move_to_next( &d_ant[threadIdx.x], step );
    }

    step = d_instance->n;

    // Connect the tour (ie end at beginning city)
    d_ant[threadIdx.x].tour[d_instance->n] = d_ant[threadIdx.x].tour[0];
    // Sums distances of all segements in tour
    d_ant[threadIdx.x].tour_length = d_compute_tour_length( d_ant[threadIdx.x].tour );
}


/** init CUDA random number generator */
__global__ 
void rand_init() {
    curand_init((unsigned long long)clock(), threadIdx.x, 0, &state);    
}

/** setup memory on GPU and copy data */
void init_device() {
    rand_init<<<n_ants, 1>>>();

    instance_for_d.n = n;
    instance_for_d.n_near = nn_ants;

    long nn = MAX(nn_ls,nn_ants);
    if ( nn >= n )
        nn = n - 1;

    instance_for_d.nn_list = (long**)malloc(sizeof(long int) * n * 
                                                nn + n * sizeof(long int *));

    // TODO setup nn_list and distance to be copied to device
    /* instance_to_device.nn_list; */
    /* instance_to_device.distance; */

    mallocOnGPU(instance_to_d, sizeof(d_problem_t));
    copyToGPU(instance_to_d, &instance_for_d, sizeof(d_problem_t));

    /* setup ants for device */
    mallocOnGPU(ant_to_d, sizeof(ant_struct) * n_ants);
    copyToGPU(ant_to_d, ant, sizeof(ant_struct) * n_ants);

    /* setup pheromone on device (size taken from generate_double_matrix */
    // TODO must setup pointers for two dimensional array
    int mat_size = sizeof(double) * n * n + sizeof(double *) * n;
    mallocOnGPU(total_to_d, mat_size);
    copyToGPU(total_to_d, total, mat_size);
}

/* --- main program ------------------------------------------------------ */

int main(int argc, char *argv[])
/*
      FUNCTION:       main control for running the ACO algorithms
      INPUT:          none
      OUTPUT:         none
      (SIDE)EFFECTS:  none
      COMMENTS:       this function controls the run of "max_tries" independent trials

*/
{
    long int i;

    start_timers();

    init_program(argc, argv);

    instance.nn_list = compute_nn_lists();
    pheromone = generate_double_matrix( n, n );
    total = generate_double_matrix( n, n );

    init_device();

    time_used = elapsed_time( REAL );
    printf("Initialization took %.10f seconds\n",time_used);

    for ( n_try = 0 ; n_try < max_tries ; n_try++ ) {

        init_try(n_try);

        while ( !termination_condition() ) {

            //construct_solutions();
            construct_solutions<<<n_ants, 1>>>(ant_to_d, instance_to_d, total_to_d);
            
            // Copy ants back for updating pheromones
            copyFromGPU(ant, ant_to_d, sizeof(ant_struct) * n_ants);

            // update n_tours
            n_tours += n_ants;

            if ( ls_flag > 0 )
                local_search(); // TODO possible to integrate GPU 2opt??

            update_statistics();

            pheromone_trail_update();

            // TODO copy new pheromones to device

            search_control_and_statistics();

            iteration++;
        }

        exit_try(n_try);
    }

    exit_program();

    free( instance.distance );
    free( instance.nn_list );
    free( pheromone );
    free( total );
    free( best_in_try );
    free( best_found_at );
    free( time_best_found );
    free( time_total_run );

    for ( i = 0 ; i < n_ants ; i++ ) {
        free( ant[i].tour );
        free( ant[i].visited );
    }

    free( ant );
    free( best_so_far_ant->tour );
    free( best_so_far_ant->visited );
    /* free( prob_of_selection ); */

    return(0);
}
