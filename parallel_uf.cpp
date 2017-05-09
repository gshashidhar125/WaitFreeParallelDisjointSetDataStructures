#include "parallel_uf.hpp"
#include "atomic_support.h"

Node_t *makeSet(long id){
    return new Node_t(id);
}


Node_t* find(Node_t* x) {

    Node_t *p = x->parent;
    Node_t *gp = p->parent;

    while(true){
        bool retry = false;
        while (p != x){
            if (!CAS((uint64_t *)&x->parent, (uint64_t)p, (uint64_t)gp)){
                retry = true;
                break;
            }
            x = x->parent;
            p = x->parent;
            gp = p->parent;
        }
        if (!retry){
            return x;
        }
    }

}

bool sameSet(Node_t* x, Node_t* y) {

    Node_t *p_x = find(x);
    Node_t *p_y = find(y);

    return (p_x == p_y);
}

void makeUnion(Node_t* x, Node_t* y) {

    while(true){
        Node_t *p_x = find(x);
        Node_t *p_y = find(y);

        uf_rank_t rank_px = p_x->rank;
        uf_rank_t rank_py = p_y->rank;

        if (rank_px > rank_py){
            /* y needs to be merged into x */
            /* No update to rank of either one */

            /* TODO:What if (p_x->parent != p_x)? */
            if (p_x == find(x)){ /* Is p_x the patent of x */
                /* TODO: There is no atomicity between the check and swing */
                if (CAS((uint64_t *)&p_y->parent, (uint64_t)p_y, (uint64_t)p_x)){
                    return;
                }
            }
        }
        else if (rank_py > rank_px){
            if (p_y == find(y)){ /* Is p_y the patent of y */
                if (CAS((uint64_t *)&p_x->parent, (uint64_t)p_x, (uint64_t)p_y)){
                    return;
                }
            }
        }
        else{
            /* rank_px == rank_py */
            /* Lets us always pick this
             *    : p_x gets merged into p_y */
            if (CAS((uint64_t *)&p_x->parent, (uint64_t)p_x, (uint64_t)p_y)){
                /* We need to update p_y's rank */
                uf_rank_t oldrank = p_y->rank;
                uf_rank_t newrank = oldrank + 1;
                CAS((uint32_t *)&p_y->rank, oldrank, newrank);
                /* Don't worry even if the CAS fails as well
                 * B/C we may have someone incremented it for us:)
                 */
                return;
            }

        }
    }
}


