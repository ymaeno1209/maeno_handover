PREFIX=/home/pj24001725/ku40000080/cotocoa-garea-bcast-pattern
CC=mpicc
NC=nvcc -ccbin g++
CFLAGS=-I${PREFIX}/include
LDFLAGS=-L${PREFIX}/lib
LIBS=-lctca

PROGS = diff_requester diff_coupler diff_worker diff_requester_new

all: ${PROGS}

diff_requester: diff_requester.c
	$(NC) $(CFLAGS) $(LDFLAGS) diff_requester.c sim_diffusion.c sim_misc.c kernel.cu util.c -lmpi -lcuda -lctca -o diff_requester

diff_coupler: diff_coupler.c
	$(CC) $(CFLAGS) $(LDFLAGS) diff_coupler.c -lctca -o diff_coupler

diff_worker: diff_worker.c
	$(CC) $(CFLAGS) $(LDFLAGS) diff_worker.c myPng.c util.c -lm -lpng -fopenmp -lctca -o diff_worker

diff_requester_new: diff_requester_new.c
	$(NC) $(CFLAGS) $(LDFLAGS) diff_requester_new.c sim_diffusion.c sim_misc.c kernel.cu util.c -lmpi -lcuda -lctca -o diff_requester_new
clean:
	/bin/rm -f *.o *.mod ${PROGS}
