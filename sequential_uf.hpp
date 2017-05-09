#ifndef _SEQUENTIAL_UF_H_
#define _SEQUENTIAL_UF_H_

#include "typedefines.h"
#include <map>


typedef struct Node {

	uint64_t id;
	struct Node* parent;
	uint64_t rank;

    Node(uint64_t p_id): id(p_id), parent(this), rank(0){

    }
} Node_t;


class sequential_uf_t{

    static std::map<uint64_t, Node_t *> mapped;
    public:
        /* desc: Creates a Node element 
         *
         * prec: The id should be unique. No makeSet on the same
         *       id should have happend before this call
         */
        static void makeSet(uint64_t id);
        


        /* Returns the parent to the node. */
        /* It can be itself, if the node is isolated 
         * or the representative node */
        static uint64_t find(uint64_t);

        static bool sameSet(uint64_t, uint64_t);

        /* Union of two sets. Higher rank node will become 
         * the representative of the union */
        static void makeUnion(uint64_t, uint64_t);

        static void printNodes();
};

#endif
