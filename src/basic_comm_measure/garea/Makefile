CC=mpicc
NC=nvcc -ccbin g++ -D_ENABLE_CUDA_
LIBS=-lcuda
PROGS = garea_write

all: ${PROGS}

garea_write: garea_write.c
	$(NC) garea_write.c garea.c osu_1sc.c $(LIBS) -o garea_write -lmpi -lcudart

clean:
	/bin/rm -f *.o *.mod ${PROGS}
