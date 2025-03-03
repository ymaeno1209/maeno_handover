PREFIX=/home/pj24001725/ku40000080/test/cotocoa
CC=mpicc
NC=nvcc -ccbin g++
CFLAGS=-I${PREFIX}/include
LDFLAGS=-L${PREFIX}/lib
LIBS=-lctca

PROGS = requester_garea_write coupler_garea_write worker1_garea_write \
		requester_garea_write_D\
		attach_test attach_test0

all: ${PROGS}

requester_garea_write: requester_garea_write.c
	$(CC) $(CFLAGS) $(LDFLAGS) requester_garea_write.c ../garea.c $(LIBS) -o requester_garea_write

coupler_garea_write: coupler_garea_write.c
	$(CC) $(CFLAGS) $(LDFLAGS) coupler_garea_write.c ../garea.c $(LIBS) -o coupler_garea_write

worker1_garea_write: worker1_garea_write.c
	$(CC) $(CFLAGS) $(LDFLAGS) worker1_garea_write.c ../garea.c $(LIBS) -o worker1_garea_write

requester_garea_write_D: requester_garea_write_D.c
	$(NC) $(CFLAGS) $(LDFLAGS) requester_garea_write_D.c ../garea.c $(LIBS) -o requester_garea_write_D -lmpi -lcudart

attach_test: attach_test.c
	$(NC) $(CFLAGS) $(LDFLAGS) attach_test.c ../garea.c $(LIBS) -o attach_test -lmpi -lcudart

attach_test0: attach_test0.c
	$(NC) $(CFLAGS) $(LDFLAGS) attach_test0.c ../garea.c $(LIBS) -o attach_test0 -lmpi -lcudart	

clean:
	/bin/rm -f *.o *.mod ${PROGS}
