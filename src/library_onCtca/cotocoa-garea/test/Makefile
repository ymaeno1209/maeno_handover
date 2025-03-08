PREFIX=/home/pj24001725/ku40000080/cotocoa-garea-bcast-pattern
CC=mpicc
FC=mpifort
CFLAGS=-I${PREFIX}/include
LDFLAGS=-L${PREFIX}/lib
LIBS=-lctca
F_LIBS=-lctca_f -lctca_mod

PROGS = requester_garea_read_copy coupler_garea_read_copy worker1_garea_read_copy \
		requester_garea_read coupler_garea_read worker1_garea_read \

all: ${PROGS}

requester_garea_read: requester_garea_read.c
	$(CC) $(CFLAGS) $(LDFLAGS) requester_garea_read.c $(LIBS) -o requester_garea_read

coupler_garea_read: coupler_garea_read.c
	$(CC) $(CFLAGS) $(LDFLAGS) coupler_garea_read.c $(LIBS) -o coupler_garea_read

worker1_garea_read: worker1_garea_read.c
	$(CC) $(CFLAGS) $(LDFLAGS) worker1_garea_read.c $(LIBS) -o worker1_garea_read

requester_garea_read_copy: requester_garea_read_copy.c
	$(CC) $(CFLAGS) $(LDFLAGS) requester_garea_read_copy.c  $(LIBS) -o requester_garea_read_copy

coupler_garea_read_copy: coupler_garea_read_copy.c
	$(CC) $(CFLAGS) $(LDFLAGS) coupler_garea_read_copy.c  $(LIBS) -o coupler_garea_read_copy

worker1_garea_read_copy: worker1_garea_read_copy.c
	$(CC) $(CFLAGS) $(LDFLAGS) worker1_garea_read_copy.c  $(LIBS) -o worker1_garea_read_copy

clean:
	/bin/rm -f *.o *.mod ${PROGS}
