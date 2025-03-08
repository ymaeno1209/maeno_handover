Installation of CoToCoA
=====

1. Download and extract a release
-----
- Latest release is available from:

  [https://github.com/tnanri/cotocoa/releases](https://github.com/tnanri/cotocoa/releases)

- Extract it with GNU tar:

  ``tar zxf cotocoa-?.?.tar.gz``

2. Modify Makefile
-----
- Go to src/ directory
- At least, modify the following lines of Makefile according to your environment:

  `PREFIX=/usr/local/ctca-1.0`

  `CC=mpicc`

  `FC=mpif90` 

  `CFLAGS=-I.`

  `LDFLAGS=-L.`

3. Compile
-----
- Run 

  ``make``

4. Install
-----
 - Run

  ``make install``



