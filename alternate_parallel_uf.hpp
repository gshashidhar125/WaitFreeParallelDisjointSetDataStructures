#ifndef _ALTERNATE_PARALLEL_UF_H_
#define _ALTERNATE_PARALLEL_UF_H_

#include "typedefines.h"
#include "atomic_support.h"
#include <assert.h>

#define USE_NUM_BITS_FOR_HASHING 20
#define USE_HASH(x) ((x) &  ((1 << USE_NUM_BITS_FOR_HASHING) - 1))
#define USE_NUM_HASH_BUCKETS (1 << USE_NUM_BITS_FOR_HASHING)


#define USE_ENTRIES_PER_NODE 3 /* 0 -> ID, 1 -> Parent, 2 -> rank */
#define USE_ENTRIES_PER_NODE_IN_LEVEL0_HASHTABLE 2 /* 0 -> index, 1-> length */
#define USE_ENTRIES_PER_NODE_IN_LEVEL1_HASHTABLE 2 /* 0 -> id, 1-> index */

class alternate_parallel_uf_t{

    private:
        static uint32_t num_nodes;
        static uint64_t *node_table;
        static uint64_t *level_0_table;
        static uint64_t *level_1_table;

    public:


        static void init( uint32_t p_num_nodes,
                          uint64_t *p_node_table, 
                          uint64_t *p_level_0_table, 
                          uint64_t *p_level_1_table 
                          ){

            num_nodes     = p_num_nodes;
            node_table    = p_node_table;
            level_0_table = p_level_0_table;
            level_1_table = p_level_1_table;
        }

        static uint32_t getIndex(uint64_t id){
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
        static uint64_t find (uint64_t id){

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


        static uint64_t getRank(uint64_t id){
            uint32_t index = getIndex(id);
            return GETRANKUSINGINDEX(index);
        }

        static bool sameSet(uint64_t x, uint64_t y){
            uint64_t p_x = find(x);
            uint64_t p_y = find(y);

            return (p_x == p_y);
        }

        static void makeUnion(uint64_t x, uint64_t y) {

            while(true){

                uint64_t p_x = find(x);
                uint64_t p_y = find(y);

                uint32_t p_x_index = getIndex(p_x);
                uint32_t p_y_index = getIndex(p_y);

                if (x < y){
                    /* y needs to be merged into x */
                    if (CAS( &node_table[p_y_index * USE_ENTRIES_PER_NODE + 1], p_y, p_x)){
                        return;
                    }
                }
                else {
                    if (CAS( &node_table[p_x_index * USE_ENTRIES_PER_NODE + 1], p_x, p_y)){
                        return;
                    }
                }
#if 0
                else{
                    /* rank_px == rank_py */
                    /* Lets us always pick this
                     *    : p_x gets merged into p_y */

                    uint64_t oldrank = GETRANKUSINGINDEX(p_y_index);
                    uint64_t newrank = oldrank + 1;
                    if (CAS( &node_table[p_x_index * USE_ENTRIES_PER_NODE + 1], p_x, p_y)){
                        /* We need to update p_y's rank */
                        CAS( &node_table[p_y_index * USE_ENTRIES_PER_NODE + 2], oldrank, newrank);
                        /* Don't worry even if the CAS fails as well
                         * B/C we may have someone incremented it for us:)
                         */
                        return;
                    }

                }
#endif
            }
        }

};

#endif
