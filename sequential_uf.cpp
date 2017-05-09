#include "sequential_uf.hpp"
#include <assert.h>
#include <iostream>
#include <list>

std::map<uint64_t, Node_t *> sequential_uf_t::mapped;


void sequential_uf_t::makeSet(uint64_t id){
    assert(mapped.find(id) == mapped.end());
    mapped[id] = new Node_t(id);
}


uint64_t sequential_uf_t::find(uint64_t id) {

    Node_t *x = mapped[id];
    Node_t *p = x->parent;

    while (p != x){

        Node_t *gp = p->parent;
        x->parent = gp;


        x = x->parent;
        p = x->parent;
    }

    return x->id;
}

bool sequential_uf_t::sameSet(uint64_t x_id, uint64_t y_id) {

    return (find(x_id) == find(y_id));
}

void sequential_uf_t::makeUnion(uint64_t x_id, uint64_t y_id) {


    while(true){
        Node_t *p_x = mapped[find(x_id)];
        Node_t *p_y = mapped[find(y_id)];

        uint64_t rank_px = p_x->rank;
        uint64_t rank_py = p_y->rank;

        if (rank_px > rank_py){
            /* y needs to be merged into x */
            /* No update to rank of either one */

            p_y->parent = p_x;
            return;
        }
        else if (rank_py > rank_px){
            p_x->parent = p_y;
            return;
        }
        else{
            /* rank_px == rank_py */
            /* Lets us always pick this
             *    : p_x gets merged into p_y */
            p_x->parent = p_y;
            p_y->rank++;
            return;
        }
    }
}

void sequential_uf_t::printNodes(){
    std::map < uint64_t, std::list<uint64_t> > connected;
    for (std::map<uint64_t, Node_t *>::iterator it = mapped.begin(); it != mapped.end(); ++it){
        uint64_t node = it->second->id;
        uint64_t parent = sequential_uf_t::find(node);
        //std::cout <<  node << ":" << parent << std::endl;
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
}


