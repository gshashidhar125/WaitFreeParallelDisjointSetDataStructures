
#include <pthread.h>
#include <iostream>
#include <map>
#include <list>


#ifdef SEQUENTIAL_CODE
#include "sequential_uf.hpp"
#define NUM_JOB_PER_THREAD num_nodes
#else
#include "alternate_parallel_uf.hpp"
#ifndef NUM_JOB_PER_THREAD
#define NUM_JOB_PER_THREAD 32*1024
#endif
#endif

typedef  struct __per_thread_info_t{
    uint32_t start;
    uint32_t len;
    __per_thread_info_t(uint32_t s, uint32_t e): start(s), len(e){
    }
}per_thread_info_t;



static uint64_t *edge_table[2];
static uint64_t *node_table;
static uint32_t num_nodes;
static uint32_t num_edges;

void makeSetCB(uint64_t id){
#ifdef SEQUENTIAL_CODE
    sequential_uf_t::makeSet(id);
#endif
}

void* do_union(void *arg){

    uint32_t start  = ((per_thread_info_t *)arg)->start;
    uint32_t length = ((per_thread_info_t *)arg)->len;
    uint32_t num = length - start;


    if (num > NUM_JOB_PER_THREAD){
        int mid = num / 2;

        pthread_t t0, t1;
        per_thread_info_t p0(start, start + mid);
        per_thread_info_t p1(start + mid, length);
        // std::cout << start << ":" << start + mid << ", " << start + mid << ":" << length << std::endl;
        pthread_create(&t0, NULL, do_union, (void *)&p0);
        pthread_create(&t1, NULL, do_union, (void *)&p1);

        void *status;
        pthread_join(t0, &status);
        pthread_join(t1, &status);
    }
    else{
        for (uint32_t i = start; i < length; ++i){
#ifdef SEQUENTIAL_CODE
            sequential_uf_t::makeUnion(edge_table[0][i], edge_table[1][i]);
#else
            alternate_parallel_uf_t::makeUnion(edge_table[0][i], edge_table[1][i]);
#endif
        }
    }
    pthread_exit(NULL);
}

void doWork(
        uint32_t p_num_nodes, uint32_t p_num_edges,
        uint64_t *p_node_table,
        uint64_t *level_0_table,
        uint64_t *level_1_table,
        uint64_t *p_edge_table[2]){


    edge_table[0] = p_edge_table[0];
    edge_table[1] = p_edge_table[1];
    node_table    = p_node_table;
    num_nodes     = p_num_nodes;
    num_edges     = p_num_edges;

#ifdef SEQUENTIAL_CODE
    std::cout << "Sequential(x86):" << std::endl;
    std::cout << " Num Jobs            : " << p_num_edges << std::endl;
#else
    std::cout << "Parallel(x86): " << std::endl;
    std::cout << " Num Jobs            : " << p_num_edges << std::endl;
    std::cout << " Num Jobs per thread : " << NUM_JOB_PER_THREAD << std::endl;
#endif


#ifndef SEQUENTIAL_CODE
    alternate_parallel_uf_t::init(num_edges, node_table, level_0_table, level_1_table);
#endif

    pthread_t t0;
    per_thread_info_t p0(0, num_edges);
    pthread_create(&t0, NULL, do_union, (void *)&p0);

    void *status;
    pthread_join(t0, &status);


}

void printResults(){

#ifdef SEQUENTIAL_CODE

    sequential_uf_t::printNodes();
#else
    std::map < uint64_t, std::list<uint64_t> > connected;
    for (uint32_t i = 0; i < num_nodes; i++){
        uint64_t node = node_table[i * USE_ENTRIES_PER_NODE + 0];
        uint64_t parent = alternate_parallel_uf_t::find(node);
        connected[parent].push_back(node);
    }
    for (std::map < uint64_t, std::list<uint64_t> > :: iterator it = connected.begin(); it != connected.end(); ++it){
        // std::cout << it->first << " : ";
        it->second.sort();
        for (std::list<uint64_t>::iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2){
            std::cout << *it2 << " ";
        }
        std::cout << std::endl;
    }
#endif
}
