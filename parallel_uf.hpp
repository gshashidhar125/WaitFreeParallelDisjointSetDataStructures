#ifndef _PARALLEL_UF_H_
#define _PARALLEL_UF_H_

#include "typedefines.h"
typedef uint32_t       uf_rank_t;

typedef struct Node {

	uint64_t id;
	struct Node* parent;
	uf_rank_t rank;

    Node(uint64_t p_id): id(p_id), parent(this){

    }

	bool operator == (const Node &in) const {		
        return (id == in.id);
    }

	bool operator < (const Node &in) const {		// To make this structure comparable
		return (id < in.id);
	}

} Node_t;


/* desc: Creates a Node element 
 *
 * prec: The id should be unique. No makeSet on the same
 *       id should have happend before this call
 */
Node_t *makeSet(uint64_t id);


/* Returns the parent to the node. */
/* It can be itself, if the node is isolated 
 * or the representative node */
Node_t* find(Node_t*);

bool sameSet(Node_t*, Node_t*);

/* Union of two sets. Higher rank node will become 
 * the representative of the union */
void takeUnion(Node_t*, Node_t*);

#endif
