

#include "alternate_parallel_uf.hpp"
#include <stdlib.h>

uint32_t  alternate_parallel_uf_t::num_nodes     = 0;
uint64_t *alternate_parallel_uf_t::node_table    = NULL;
uint64_t *alternate_parallel_uf_t::level_0_table = NULL;
uint64_t *alternate_parallel_uf_t::level_1_table = NULL;
