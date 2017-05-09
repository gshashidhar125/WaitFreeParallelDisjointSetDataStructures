

#include <iostream>
#include <fstream>
#include "typedefines.h"
#include <list>
#include <map>
#include <set>
#include <sys/time.h>
#include <stdlib.h>

#define NUM_BITS_FOR_HASHING 20
#define HASH(x) ((x) &  ((1 << NUM_BITS_FOR_HASHING) - 1))
#define NUM_HASH_BUCKETS (1 << NUM_BITS_FOR_HASHING)

#define ENTRIES_PER_NODE 3 /* 0 -> ID, 1 -> Parent, 2 -> rank */
#define ENTRIES_PER_NODE_IN_LEVEL0_HASHTABLE 2 /* 0 -> index, 1-> length */
#define ENTRIES_PER_NODE_IN_LEVEL1_HASHTABLE 2 /* 0 -> id, 1-> index */



static std::set<uint64_t> nodes;
// static std::list<std::pair<uint64_t, uint64_t> > edges;
static uint64_t *node_table;
static uint64_t *edge_table[2];

static struct timeval t0, t1;

void makeSetCB(uint64_t);
void printResults();
void doWork(
        uint32_t num_nodes, uint32_t num_edges,
        uint64_t *node_table,
        uint64_t *level_0_table,
        uint64_t *level_1_table,
        uint64_t *edge_table[2]);

void printUsage(std::string exec_name){
    std::cout << "Usage:" << exec_name << " <Name of the file containing graph> <num edges to process>" << std::endl;
}


bool processFile(std::string filename, uint32_t& num_edges_to_process){

    uint64_t v0 = 0;
    uint64_t v1 = 0;

    std::ifstream myfile (filename.c_str(), std::ifstream::in);
    if (myfile.is_open())     
    {   
        uint32_t i = 0;

        edge_table[0] = new uint64_t[num_edges_to_process];
        edge_table[1] = new uint64_t[num_edges_to_process];
        while ((!myfile.eof()) && (i < num_edges_to_process))
        {
            myfile>>v0>>v1;
            nodes.insert(v0);
            nodes.insert(v1);
            //edges.push_back(std::make_pair<uint64_t, uint64_t>(v0, v1));
            edge_table[0][i] = v0;
            edge_table[1][i] = v1;

            i++;
        }
        num_edges_to_process = i;
        myfile.close();
        return true;
    }
    else{
        return false;
    }
}



void prepareTables(uint32_t num_edges_to_process){

    uint32_t num_nodes = nodes.size();

    std::map<uint32_t, std::list<std::pair<uint64_t, uint32_t> > > hash_table;


    node_table = new uint64_t[num_nodes * ENTRIES_PER_NODE];

    uint32_t i = 0;
    uint64_t rank = 0;
    for (std::set<uint64_t>::iterator it = nodes.begin(); it != nodes.end(); ++it, ++i){
        uint64_t id = *it;
        node_table[i * ENTRIES_PER_NODE + 0] = id;
        node_table[i * ENTRIES_PER_NODE + 1] = id;
        node_table[i * ENTRIES_PER_NODE + 2] = rank;
        /* <id, i> Need to be inserted into the hash */
        uint32_t hash_key = HASH(id);

        hash_table[hash_key].push_back(std::make_pair<uint64_t, uint32_t>(id, i));

        makeSetCB(id);
    }

    /* Prepare the hash-tables */
    uint64_t *level_0_table  = new uint64_t[ ENTRIES_PER_NODE_IN_LEVEL0_HASHTABLE * NUM_HASH_BUCKETS];
    uint64_t *level_1_table  = new uint64_t[ ENTRIES_PER_NODE_IN_LEVEL1_HASHTABLE * num_nodes];

    uint32_t indexes = 0;
    uint32_t entry   = 0;
    for (i = 0; i < NUM_HASH_BUCKETS; i++){
        //std::cout << i << ":" << indexes << ":" << hash_table[i].size() << std::endl;
        level_0_table[i * ENTRIES_PER_NODE_IN_LEVEL0_HASHTABLE + 0] = indexes;
        level_0_table[i * ENTRIES_PER_NODE_IN_LEVEL0_HASHTABLE + 1] = hash_table[i].size();
        indexes += hash_table[i].size();

        for (std::list<std::pair<uint64_t, uint32_t> > ::iterator it2 = hash_table[i].begin();
                it2 != hash_table[i].end(); ++it2, entry++){
            level_1_table[entry * ENTRIES_PER_NODE_IN_LEVEL1_HASHTABLE + 0] = it2->first;
            level_1_table[entry * ENTRIES_PER_NODE_IN_LEVEL1_HASHTABLE + 1] = it2->second;
        }
    }

#if 0
    /* prepare the edge tables */
    edge_table[0] = new uint64_t[edges.size()];
    edge_table[1] = new uint64_t[edges.size()];

    i = 0;
    for (std::list<std::pair<uint64_t, uint64_t> >::iterator it3 = edges.begin(); it3 != edges.end(); ++it3, ++i){
        edge_table[0][i] = it3->first;
        edge_table[1][i] = it3->second;
    }
#endif

	gettimeofday(&t0, 0);

    doWork(num_nodes, num_edges_to_process,
            node_table,
            level_0_table,
            level_1_table,
            edge_table);

	gettimeofday(&t1, 0);
	long elapsed = (t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec;
	std::cout << " Time Taken          : " << elapsed << " us" << std::endl;
    

    printResults();

}

int main(int argc, char **argv){

    std::string filename;
    uint32_t num_edges_to_process = 0;

    bool filename_found = false;
    for (int i = 1; i < argc; i++){
        switch (i){
            case 1:
                filename_found = true;
                filename.assign(argv[i]);
                break;
            case 2:
                num_edges_to_process = atoi(argv[i]);
                break;
            default:
                break;
        }
    }

    if (!filename_found){
        printUsage(argv[0]);
        return -1;
    }

    if (num_edges_to_process == 0){
        num_edges_to_process = 2*1024*1024; /* A million edge max */
    }


    if (!processFile(filename, num_edges_to_process)){
        return -1;
    }

    prepareTables(num_edges_to_process);

    return 0;
}
