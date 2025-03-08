### 概要
basic_comm_measrueでは通信の基本性能を計測するためのコードを示す。

### 内容
以下の題名はソースコードで用いたディレクトリ名と対応している。
- buffer_interface：本研究で実装したバッファインターフェースの計測
- garea：既存のCoToCoAで全てのサブコミュニケータがwindowを登録し、片側通信が行える関数群であるgareaの計測
- gdcopy_test：GPUDirect Copyの有無による計測
- osu-micro-benchmark/mpi/pt2pt：両側通信であるMPI_Send/MPI_Recvのノード間又はノード内で計測
- osu-micro-benchmark/mpi/one-sided：片側通信であるMPI_Put・MPI_Getのノード間又はノード内で計測

### プログラムの実行方法
1. module load gcc/8 hpcx/2.17.1 cuda/12.2.2
2. make clean
3. make -j 
4. ジョブを投げる