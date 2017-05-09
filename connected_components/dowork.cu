#include "cuda_parallel_uf.cu"



void makeSetCB(uint64_t id){

}


void printResults(uint64_t * node_table, uint32_t num_nodes, uint32_t num_edges,
        uint64_t * level_0_table, uint64_t * level_1_table);


#ifndef MAX_JOB_PER_THREAD
#define MAX_JOB_PER_THREAD 64
#endif

#ifndef NUM_THREADS_PER_BLOCK
#define NUM_THREADS_PER_BLOCK 1024
#endif

__global__ void do_union(){

        uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t len = MAX_JOB_PER_THREAD;
        //uint32_t len = 1;
        index = index * len;

        for (uint32_t i = index; i < index + len; ++i){
                if (i < num_edges){
                        print("Thread: %d. Edge(%ld, %ld)\n", index, edge_table0[i], edge_table1[i]);
                        makeUnion(edge_table0[i], edge_table1[i]);
                }
        }
}

void doWork(
        uint32_t p_num_nodes, uint32_t p_num_edges,
        uint64_t *p_node_table,
        uint64_t *level_0_table,
        uint64_t *level_1_table,
        uint64_t *p_edge_table[2]){


    init(p_num_nodes, p_num_edges, p_node_table, level_0_table, level_1_table, p_edge_table);


    uint32_t NUM_BLOCKS  = (p_num_edges - 1)/(NUM_THREADS_PER_BLOCK * MAX_JOB_PER_THREAD) + 1;
    do_union<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>();
    cudaThreadSynchronize();

    uint64_t *d_a;
    print("Hereh1: %d\n", d_a);
    cudaError_t err;
    cudaMemcpyFromSymbol(&d_a, node_table, sizeof(uint64_t *), 0, cudaMemcpyDeviceToHost);
    print("Hereh1: %d\n", d_a);
    err = cudaMemcpy(p_node_table, d_a, p_num_nodes * sizeof(Node), cudaMemcpyDeviceToHost);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
         //     return EXIT_FAILURE;
    }
    cudaThreadSynchronize();
    //printResults(p_node_table, p_num_nodes, p_num_edges, level_0_table, level_1_table);
}

