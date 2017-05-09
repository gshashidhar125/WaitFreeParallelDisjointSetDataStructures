#include "alternate_parallel_uf.hpp"
#include <iostream>
#include <map>
#include <list>
#include <stdlib.h>
#include <stdio.h>


void printResults(){

}
void printResults(uint64_t * p_node_table, uint32_t p_num_nodes, uint32_t p_num_edges, 
        uint64_t* p_level_0_table, uint64_t* p_level_1_table){

    alternate_parallel_uf_t::init(p_num_edges, p_node_table, p_level_0_table,
        p_level_1_table);
    std::map < uint64_t, std::list<uint64_t> > connected;
    for (uint32_t i = 0; i < p_num_nodes; i++){
        uint64_t node = p_node_table[i * USE_ENTRIES_PER_NODE + 0];
        uint64_t parent = alternate_parallel_uf_t::find(node);
        connected[parent].push_back(node);
//        printf("Node[%ld], Parent = %ld, rank = %ld\n", node, p_node_table[i * USE_ENTRIES_PER_NODE + 1], 
//                p_node_table[i * USE_ENTRIES_PER_NODE + 2]);
    }

    for (std::map < uint64_t, std::list<uint64_t> > :: iterator it = connected.begin(); it != connected.end(); ++it){
        it->second.sort();
        for (std::list<uint64_t>::iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2){
            std::cout << *it2 << " ";
        }
        std::cout << std::endl;
    }
}
