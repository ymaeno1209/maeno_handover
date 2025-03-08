PREFIX=../../library_onCtca/cotocoa-garea
CC=mpicc
NC=nvcc -ccbin g++ -D_ENABLE_CUDA_
CFLAGS=-I${PREFIX}/include
LDFLAGS=-L${PREFIX}/lib
LIBS=-lctca -lm

PROGS = buffer_write \
		buffer_write_req buffer_write_cpl buffer_write_wrk \
		test_req test_cpl test_wrk

all: ${PROGS}


buffer_write:
	$(NC) $(CFLAGS) $(LDFLAGS) buffer_write.c buffer.c buffer_kernel.cu $(LIBS) -o buffer_write -lmpi -lcudart
buffer_write_req:
	$(NC) $(CFLAGS) $(LDFLAGS) buffer_write_req.c buffer.c buffer_kernel.cu $(LIBS) -o buffer_write_req -lmpi -lcudart
buffer_write_cpl:
	$(CC) $(CFLAGS) $(LDFLAGS) buffer_write_cpl.c buffer.c $(LIBS) -o buffer_write_cpl
buffer_write_wrk:
	$(CC) $(CFLAGS) $(LDFLAGS) buffer_write_wrk.c buffer.c $(LIBS) -o buffer_write_wrk

test_req:
	$(NC) $(CFLAGS) $(LDFLAGS) test_req.c buffer.c buffer_kernel.cu $(LIBS) -o test_req -lmpi -lcudart
test_cpl:
	$(CC) $(CFLAGS) $(LDFLAGS) test_cpl.c buffer.c $(LIBS) -o test_cpl
test_wrk:
	$(CC) $(CFLAGS) $(LDFLAGS) test_wrk.c buffer.c $(LIBS) -o test_wrk

clean:
	/bin/rm -f *.o *.mod ${PROGS}
