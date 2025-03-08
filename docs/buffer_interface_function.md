### 概要
本研究で実装したバッファインターフェースの関数群について、それぞれの関数について示す。

### 公開している関数一覧

Requesterの整数型のバッファインターフェース初期化関数
- void requester_buffer_init_withint(int *requester_data_division, size_t datasize, bool data_direction);

Requesterの単精度のバッファインターフェース初期化関数
- void requester_buffer_init_withreal4(int *requester_data_division, size_t datasize, bool data_direction);

Requesterの倍精度のバッファインターフェース初期化関数
- void requester_buffer_init_withreal8(int *requester_data_division, size_t datasize, bool data_direction);

Requesterの書き込み関数
- void requester_buffer_write(void *src_data, int step);

Requesterのバッファインターフェース終了関数
- void requester_buffer_fin();

Workerの整数型のバッファインターフェース初期化関数
- void worker_buffer_init_withint(int *worker_data_division, target_side_type target_side, int *worker_data_oneside_length);

Workerの単精度のバッファインターフェース初期化関数
- void worker_buffer_init_withreal4(int *worker_data_division, target_side_type target_side, int *worker_data_oneside_length);

Workerの倍精度のバッファインターフェース初期化関数
- void worker_buffer_init_withreal8(int *worker_data_division, target_side_type target_side, int *worker_data_oneside_length);

Workerの読み込み関数
- void worker_buffer_read(void *dest_data, int step);

Workerのバッファインタフェースの終了関数
- void worker_buffer_fin();

Couplerのバッファインタフェースの初期化関数
- void coupler_buffer_init();

Couplerのバッファインタフェースの終了関数
- void coupler_buffer_fin();

### 非公開関数の一覧
Requesterの任意の型で行うバッファインターフェース初期化関数
- void requester_buffer_init(int *requester_data_division, size_t datasize, bool data_direction, int type);

Workerの任意の型で行うバッファインターフェース初期化関数
- void worker_buffer_init(int *worker_data_division, target_side_type target_side, int *worker_data_oneside_length, int type);

WorkerがRequesterに送る初期化データを送るためにデータを整理する貯めの関数
- int *worker_collect_init_info(int *worker_data_division, target_side_type target_side);

MPIのデータタイプを決定する関数
- MPI_Datatype decide_MPI_datatype(int type);

MPIの単一のデータ数を決定する関数
- int decide_unitsize(int type);

garea_writeするデータの型により関数名を変え、実行するための関数。また、garea_writeを引数であるcon_packing_count回数分行う。
- void garea_write_autotype(int real_buffer_gareaid, int *target_world_rank, size_t *offset, size_t *packed_data_element, void *comm_data, int con_packing_count);

RequesterがWorkerから初期化データを受け取るための関数
- void fetch_worker_init_info(int *worker_init_info);

WorkerがRequesterから初期化データを受け取るための関数
- void fetch_requester_init_info(int *requester_init_info);

Requesterが初期化を行う関数
- void setup_conditions(int *worker_init_info, int *requester_data_division);

Requesterのある座標からWorkerのサブコミュニケータ内でランクを返す関数
- int to_worker_localrank_from_coordinate(int global_x, int global_y, int global_z);

Requesterのある座標からWorkerのサブコミュニケータ内でオフセットを返す関数
- size_t to_offset_from_coordinate(int global_x, int global_y, int global_z);

ランク内のX座標から全てのランクから観測したX座標へと変換する関数
- void to_global_coordinate_from_local_coordinate_x(int local_x, int *global_x);

ランク内のY座標から全てのランクから観測したY座標へと変換する関数
- void to_global_coordinate_from_local_coordinate_y(int local_y, int *global_y);

ランク内のZ座標から全てのランクから観測したZ座標へと変換する関数
- void to_global_coordinate_from_local_coordinate_z(int local_z, int *global_z);

RequesterがWorkerのバッファスロットに空が出来るまで待機を行う。また、空が出来た際は何番目のスロットかを返す関数。
- int requester_polling_header_buffer();

WorkerがRequesterから送られたデータがバッファスロット内に存在するかを確認し、無ければ待機する。また存在すれば、バッファスロットの何番目に格納されているかを返す関数。
- int worker_polling_header_buffer(int step);

Requesterがheaderバッファにステップ番号を書き込む関数
- void write_header_buffer(int buffer_index, int step);

Requesterがバッファに実際のデータを書き込む関数
- void requester_write_onestep_to_realbuffer(void *src_data, int buffer_index);

バッファを管理するためのheaderバッファを初期化する関数。
- void initialize_header_buffer();

実際のデータが書き込まれるバッファを初期化する関数。
- void initialize_real_buffer();

Workerがバッファからデータを読み込んだ際に、そのバッファから読み込み終わったことを知らせバッファを空にするための関数。
- void clear_head_buffer_at_index(int buffer_index);

Requesterがバッファに書き込みを行うためのデータを初期化するための関数。どの座標がどのランクのどれくらいのオフセットに送るかを決定する。
- void initialize_requester_write_info();

Requesterがデータの書き込みを行う際に、何回同時に書き込みを行うかを決定する関数
- void initialize_con_packing_count();

RequesterがWorkerに送る初期データの整形を行う関数
- void reshape_requester_write_info();

アドレスを交換
- void swap_data_addr(void **data0, void **data1);

倍精度型のアドレスを交換。
- void swap_data_real8(size_t **data0, size_t **data1);

整数型のアドレスを交換。
- void swap_data_int(int **data0, int **data1);


### デバッグ用の関数

headerバッファに存在するステップ番号を表示する。
- void show_header_buffer();

バッファスロットの何番目が何回利用されているかをカウントする配列を初期化する関数
- void initialize_count_buffer_index();

バッファスロットの何番目が何回利用されているかをカウントする関数
- void count_buffer_index();

Requesterのデータ書き込み時間の詳細を出力する関数。
- void requester_print_detail_write_time();

Workerのデータ読み込み時間の詳細を出力する関数
- void worker_print_detail_read_time();