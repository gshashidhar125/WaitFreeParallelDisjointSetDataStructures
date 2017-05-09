#ifndef _CUDA_PARALLEL_UF_H_
#define _CUDA_PARALLEL_UF_H_

#include "typedefines.h"
#include "atomic_support.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define USE_NUM_BITS_FOR_HASHING 20
#define USE_HASH(x) ((x) &  ((1 << USE_NUM_BITS_FOR_HASHING) - 1))
#define USE_NUM_HASH_BUCKETS (1 << USE_NUM_BITS_FOR_HASHING)


#define USE_ENTRIES_PER_NODE 3 /* 0 -> ID, 1 -> Parent, 2 -> rank */
#define USE_ENTRIES_PER_NODE_IN_LEVEL0_HASHTABLE 2 /* 0 -> index, 1-> length */
#define USE_ENTRIES_PER_NODE_IN_LEVEL1_HASHTABLE 2 /* 0 -> id, 1-> index */

//#define DEBUG
#ifdef DEBUG
#define print(...) printf(__VA_ARGS__)
#else
#define print(...) ;
#endif
struct Map {
    uint64_t ID1;
    uint64_t ID2;
};
struct Node {
    uint64_t id;
    uint64_t parent;
    uint64_t rank;
};
__device__ uint32_t num_nodes;
__device__ uint32_t num_edges;
__device__ uint64_t *node_table;
__device__ uint64_t *level_0_table;
__device__ uint64_t *level_1_table;
__device__ uint64_t *edge_table0;
__device__ uint64_t *edge_table1;

void init( uint32_t p_num_nodes,
        uint32_t p_num_edges,
        uint64_t *p_node_table, 
        uint64_t *p_level_0_table, 
        uint64_t *p_level_1_table, 
        uint64_t *p_edge_table[2] 
        ){

    cudaError_t err;
    uint64_t * temp_node_table, * temp_level_0_table, * temp_level_1_table, *temp_edge_table0, *temp_edge_table1;
    cudaMemcpyToSymbol(num_nodes, &p_num_nodes, sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(num_edges, &p_num_edges, sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
    err = cudaMalloc((void **)&temp_node_table, p_num_nodes * sizeof(Node));
    if( err != cudaSuccess)
    {
        printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
        //     return EXIT_FAILURE;
    }
    err = cudaMalloc((void **)&temp_level_0_table, USE_NUM_HASH_BUCKETS * sizeof(Map));
    if( err != cudaSuccess)
    {
        printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
        //     return EXIT_FAILURE;
    }
    err = cudaMalloc((void **)&temp_level_1_table, p_num_nodes * sizeof(Map));
    if( err != cudaSuccess)
    {
        printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
        //     return EXIT_FAILURE;
    }
    err = cudaMalloc((void **)&temp_edge_table0, p_num_edges * sizeof(uint64_t));
    if( err != cudaSuccess)
    {
        printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
        //     return EXIT_FAILURE;
    }
    err = cudaMalloc((void **)&temp_edge_table1, p_num_edges * sizeof(uint64_t));
    if( err != cudaSuccess)
    {
        printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
        //     return EXIT_FAILURE;
    }
    err = cudaMemcpy(temp_node_table, p_node_table, p_num_nodes * sizeof(Node), cudaMemcpyHostToDevice);
    if( err != cudaSuccess)
    {
        printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
        //     return EXIT_FAILURE;
    }
    err = cudaMemcpy(temp_level_0_table, p_level_0_table, USE_NUM_HASH_BUCKETS * sizeof(Map), cudaMemcpyHostToDevice);
    if( err != cudaSuccess)
    {
        printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
        //     return EXIT_FAILURE;
    }
    err = cudaMemcpy(temp_level_1_table, p_level_1_table, p_num_nodes * sizeof(Map), cudaMemcpyHostToDevice);
    if( err != cudaSuccess)
    {
        printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
        //     return EXIT_FAILURE;
    }
    err = cudaMemcpy(temp_edge_table0, p_edge_table[0], p_num_edges * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if( err != cudaSuccess)
    {
        printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
        //     return EXIT_FAILURE;
    }
    err = cudaMemcpy(temp_edge_table1, p_edge_table[1], p_num_edges * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if( err != cudaSuccess)
    {
        printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
        //     return EXIT_FAILURE;
    }
    cudaMemcpyToSymbol(node_table, &temp_node_table, sizeof(uint64_t *), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(level_0_table, &temp_level_0_table, sizeof(uint64_t *), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(level_1_table, &temp_level_1_table, sizeof(uint64_t *), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(edge_table0, &temp_edge_table0, sizeof(uint64_t *), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(edge_table1, &temp_edge_table1, sizeof(uint64_t *), 0, cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
}

__device__ uint32_t getIndex(uint64_t id){
    uint32_t key = USE_HASH(id);

    uint32_t index = level_0_table[key * USE_ENTRIES_PER_NODE_IN_LEVEL0_HASHTABLE + 0];
    uint32_t len   = level_0_table[key * USE_ENTRIES_PER_NODE_IN_LEVEL0_HASHTABLE + 1];

    for (uint32_t i = index; i < index + len; i++){
        if (level_1_table[i * USE_ENTRIES_PER_NODE_IN_LEVEL1_HASHTABLE + 0] == id){
            return level_1_table[i * USE_ENTRIES_PER_NODE_IN_LEVEL1_HASHTABLE + 1];
        }
    }

    assert(0);
    return 0;
}



#define GETPARENTUSINGINDEX(index) node_table[index * USE_ENTRIES_PER_NODE + 1]
#define GETRANKUSINGINDEX(index)   node_table[index * USE_ENTRIES_PER_NODE + 2]

/* Find the representative of the given ID */
__device__  uint64_t find (uint64_t id){
    uint32_t index   = getIndex(id);
    uint64_t parent  = GETPARENTUSINGINDEX(index);

    while (parent != id){


        uint64_t pindex  = getIndex(parent);
        uint64_t gparent = GETPARENTUSINGINDEX(pindex);

        CAS(&node_table[index * USE_ENTRIES_PER_NODE + 1], parent, gparent);

        id = parent;


        index  = getIndex(id);
        parent = GETPARENTUSINGINDEX(index);

    }
    return id;
}


__device__ uint64_t getRank(uint64_t id){
    uint32_t index = getIndex(id);
    return GETRANKUSINGINDEX(index);
}

__device__ bool sameSet(uint64_t x, uint64_t y){
    uint64_t p_x = find(x);
    uint64_t p_y = find(y);

    return (p_x == p_y);
}

__device__ void makeUnion(uint64_t x, uint64_t y) {

    while(true){

        uint64_t p_x = find(x);
        uint64_t p_y = find(y);

        uint32_t p_x_index = getIndex(p_x);
        uint32_t p_y_index = getIndex(p_y);

        if (x < y){
            /* y needs to be merged into x */
            /* No update to rank of either one */
            if (CAS( &node_table[p_y_index * USE_ENTRIES_PER_NODE + 1], p_y, p_x)){
                print("Here1\n");
                return;
            }
        }
        else {
            if (CAS( &node_table[p_x_index * USE_ENTRIES_PER_NODE + 1], p_x, p_y)){
                print("Here2\n");
                return;
            }
        }
    }
}

#endif
