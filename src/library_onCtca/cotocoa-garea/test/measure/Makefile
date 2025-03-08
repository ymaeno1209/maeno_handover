PREFIX=/home/pj24001725/ku40000080/cotocoa-garea-bcast-pattern
CC=mpicc
NC=nvcc -ccbin g++ 
CFLAGS=-I${PREFIX}/include
LDFLAGS=-L${PREFIX}/lib
LIBS=-lctca

PROGS = put garea_write \
		get garea_read

all: ${PROGS}

put: 
	$(NC) put.c -o put -lmpi -lcudart
garea_write:
	$(NC) $(CFLAGS) $(LDFLAGS) garea_write.c $(LIBS) -o garea_write -lmpi -lcudart
get:
	$(NC) get.c -o get -lmpi -lcudart
garea_read:
	$(NC) $(CFLAGS) $(LDFLAGS) garea_read.c $(LIBS) -o garea_read -lmpi -lcudart

clean:
	/bin/rm -f *.o *.mod ${PROGS}
