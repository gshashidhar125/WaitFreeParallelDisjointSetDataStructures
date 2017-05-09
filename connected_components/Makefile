#DEFS=-DSEQUENTIAL_CODE
DEFS=
CFLAGS = -g $(DEFS) -o3 -Wall 
CXXFLAGS = -g $(DEFS) -o3 -Wall -I. -I/usr/include/glib-2.0/  -I/usr/lib/x86_64-linux-gnu/glib-2.0/include/ -I../

objects = dowork.o main.o alternate_parallel_uf.o sequential_uf.o

test : $(objects)
	$(CXX) -o $@ $(objects) -pthread

clean :
	rm -f test $(objects) 

