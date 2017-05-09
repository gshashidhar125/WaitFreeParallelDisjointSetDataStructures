
How to build for x86
====================

/* Sequential node-based implementation */
cd connected_components
make -f Makefile clean
make -f Makefile DEFS=-DSEQUENTIAL_CODE 


/* Sequential table implementation */
cd connected_components
make -f Makefile clean
make -f Makefile DEFS=-DNUM_JOB_PER_THREAD=num_edges

/*Parallel implementation - 32K jobs per thread*/
cd connected_components
make -f Makefile clean
make -f Makefile DEFS=-DNUM_JOB_PER_THREAD=32*1024



How to build for cuda (Need to login into libra machines)
=====================

/*Parallel implementation - 32 jobs per thread (defaults is 64 jobs per thread) */
cd connected_components
make -f Makefile.cuda clean
make -f Makefile.cuda ADDN_DEFS=-DMAX_JOB_PER_THREAD=32


/*Parallel implementation - 1024 jobs per thread (defaults is 64 jobs per thread) */
cd connected_components
make -f Makefile.cuda clean
make -f Makefile.cuda ADDN_DEFS=-DMAX_JOB_PER_THREAD=1024


