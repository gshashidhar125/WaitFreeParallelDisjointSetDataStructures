#ifndef _ATOMIC_SUPPORT_H_
#define _ATOMIC_SUPPORT_H_

#ifdef CUDA_IMPL
#define CAS(ptr,oldval,newval) atomicCAS((unsigned long long int *)ptr, \
        (unsigned long long int) oldval, (unsigned long long int)newval) == oldval
#else
#define CAS(ptr,oldval,newval) __sync_bool_compare_and_swap(ptr, oldval, newval)
#endif
#endif
