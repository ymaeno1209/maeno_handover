### 概要
library_onCtcaでは、連成フレームワークであるCoToCoAへ実装したgarea関数群とバッファインターフェース関数群を実装したソースコードとテストプログラムを示す。

### 内容
以下の題名はソースコードで用いたディレクトリ名と対応している。
- cotocoa：既存のCoToCoA
- cotocoa-garea：既存のCoToCoAにgarea関数群をマージしたコード。本研究で実装した連成計算では、このcotcoa-gareaを用いてコンパイルしている。
- garea_test：既存のCoToCoAに全てのサブコミュニケータがwindowを作成し、通信が行える関数群であるgareaのソースコードとテストコード。
- gbuffer_test：本研究で実装したバッファインターフェースのソースコードとテストコード。