CC=mpicc
NC=nvcc -ccbin g++ 
LIBS=-lm

PROGS = put_latency_test put_latency_g2c_test gpu_memory_win_create

all: ${PROGS}

put_latency_test:
	$(CC) put_latency_test.c -o put_latency_test

put_latency_g2c_test: 
	$(NC) put_latency_g2c_test.c -o put_latency_g2c_test -lmpi -lcudart
gpu_memory_win_create:
	$(NC) gpu_memory_win_create.c -o gpu_memory_win_create -lmpi -lcudart
	

clean:
	/bin/rm -f *.o *.mod ${PROGS}
