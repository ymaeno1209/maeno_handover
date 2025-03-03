PREFIX=/home/pj24001725/ku40000080/test/cotocoa
CC=mpicc
NC=nvcc -ccbin g++
CFLAGS=-I${PREFIX}/include
LDFLAGS=-L${PREFIX}/lib
LIBS=-lctca
F_LIBS=-lctca_f -lctca_mod

PROGS = requester coupler worker1 \
		requester_garea_read coupler_garea_read worker1_garea_read \
		dynamic_allocate dynamic_allocate_D

all: ${PROGS}

requester: requester.c
	$(CC) $(CFLAGS) $(LDFLAGS) requester.c garea.c $(LIBS) -o requester

coupler: coupler.c
	$(CC) $(CFLAGS) $(LDFLAGS) coupler.c garea.c $(LIBS) -o coupler

worker1: worker1.c
	$(CC) $(CFLAGS) $(LDFLAGS) worker1.c garea.c $(LIBS) -o worker1

requester_garea_read: requester_garea_read.c
	$(CC) $(CFLAGS) $(LDFLAGS) requester_garea_read.c garea.c $(LIBS) -o requester_garea_read

coupler_garea_read: coupler_garea_read.c
	$(CC) $(CFLAGS) $(LDFLAGS) coupler_garea_read.c garea.c $(LIBS) -o coupler_garea_read

worker1_garea_read: worker1_garea_read.c
	$(CC) $(CFLAGS) $(LDFLAGS) worker1_garea_read.c garea.c $(LIBS) -o worker1_garea_read

dynamic_allocate: dynamic_allocate.c
	$(CC) $(CFLAGS) $(LDFLAGS) dynamic_allocate.c garea.c $(LIBS) -o dynamic_allocate

dynamic_allocate_D: dynamic_allocate_D.c
	$(NC) $(CFLAGS) $(LDFLAGS) dynamic_allocate_D.c garea.c $(LIBS) -o dynamic_allocate_D -lmpi -lcudart

clean:
	/bin/rm -f *.o *.mod ${PROGS}
