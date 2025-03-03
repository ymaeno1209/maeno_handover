#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // 各プロセスのためのメモリ領域を確保
    int *shared_mem = (int *)malloc(sizeof(int));
    *shared_mem = rank; // 各プロセスが自分のランクを格納

    // ウィンドウを作成
    MPI_Win win;
    // MPI_Win_create(shared_mem, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    // 確保したメモリ領域をウィンドウにアタッチ
    MPI_Win_attach(win, shared_mem, sizeof(int));

    // ベースアドレスを取得
    MPI_Aint base_address;
    MPI_Get_address(shared_mem, &base_address);

    printf("rank=%d shared_mem address: %p\n", rank, (void *)shared_mem);
    printf("rank=%d base_address from MPI_Get_address: %p\n", rank, (void *)base_address);

    // 各プロセスのbase_addressを他のプロセスと共有するための配列
    MPI_Aint *base_addresses = (MPI_Aint *)malloc(nprocs * sizeof(MPI_Aint));

    // 各プロセスのアドレスを集める
    MPI_Allgather(&base_address, 1, MPI_AINT, base_addresses, 1, MPI_AINT, MPI_COMM_WORLD);

    // 同期ポイント
    MPI_Win_lock_all(0, win);

    // プロセス0がプロセス1のメモリに値を送る
    if (rank == 0)
    {
        int value; // プロセス0がプロセス1に送る値
        for (int i = 0; i < nprocs; i++)
        {
            value = 100 + i;
            MPI_Put(&value, 1, MPI_INT, i, base_addresses[i], 1, MPI_INT, win);
            // MPI_Put(&value, 1, MPI_INT, i, MPI_BOTTOM, 1, MPI_INT, win);//MPI_BOTTOMで相対アドレスを使い通信しようとしてプロセスごとに仮想アドレスの配置が異なるため想定通りのアドレス指定ができない
        }
    }

    // 同期ポイント
    MPI_Win_unlock_all(win);

    MPI_Win_detach(win, shared_mem);

    // 各プロセスが結果を表示
    printf("Rank %d has shared_mem = %d\n", rank, *shared_mem);

    // メモリの解放とウィンドウの解放
    MPI_Win_free(&win);
    free(shared_mem);

    printf("myrank %d: finalize\n", rank);

    MPI_Finalize();
    return 0;
}
