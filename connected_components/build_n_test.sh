#!/bin/bash 

MAX_EDGE_SIZE=13000000
GRAPHFILES=../../tests/*.txt

function testall {

   for graph_file in $GRAPHFILES
   do
       for i in {1..5}
       do
           echo "$i : ./test $graph_file $MAX_EDGE_SIZE"
           #./test $graph_file $MAX_EDGE_SIZE
       done
   done

}


echo "Sequential: Node based"
make clean > /tmp/_build.log
make DEFS=-DSEQUENTIAL_CODE >> /tmp/_build.log
testall

echo "Sequential: Table based"
make clean >> /tmp/_build.log
make DEFS=-DNUM_JOB_PER_THREAD=num_edges >> /tmp/_build.log
testall


echo "Parallel  : Table based (256)"
make clean >> /tmp/_build.log
make DEFS=-DNUM_JOB_PER_THREAD=256 >> /tmp/_build.log
testall

echo "Parallel  : Table based (512)"
make clean >> /tmp/_build.log
make DEFS=-DNUM_JOB_PER_THREAD=512 >> /tmp/_build.log
testall

echo "Parallel  : Table based (1*1024)"
make clean >> /tmp/_build.log
make DEFS=-DNUM_JOB_PER_THREAD=1*1024 >> /tmp/_build.log
testall

echo "Parallel  : Table based (2*1024)"
make clean >> /tmp/_build.log
make DEFS=-DNUM_JOB_PER_THREAD=2*1024 >> /tmp/_build.log
testall

echo "Parallel  : Table based (4*1024)"
make clean >> /tmp/_build.log
make DEFS=-DNUM_JOB_PER_THREAD=4*1024 >> /tmp/_build.log
testall

echo "Parallel  : Table based (8*1024)"
make clean >> /tmp/_build.log
make DEFS=-DNUM_JOB_PER_THREAD=8*1024 >> /tmp/_build.log
testall


echo "Parallel  : Table based (16*1024)"
make clean >> /tmp/_build.log
make DEFS=-DNUM_JOB_PER_THREAD=16*1024 >> /tmp/_build.log
testall

echo "Parallel  : Table based (32*1024)"
make clean >> /tmp/_build.log
make DEFS=-DNUM_JOB_PER_THREAD=32*1024 >> /tmp/_build.log
testall

