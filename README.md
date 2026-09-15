# Wasserstein holography: research notes and verification code

確率分布の輸送距離、情報幾何、量子状態の幾何が、ホログラフィーにおいて何を決めるかを検討するリポジトリです。入力式、厳密な恒等式、近似、数値結果、未検証の重力的解釈を区別します。

## 研究ノート

- 距離行列・確率流・Fisher計量・Berry曲率：[日本語PDF](docs/note.pdf)、[LaTeX](notes/note.tex)。
- 有限温度のchord分布と二境界間の測地線長：[追補PDF](docs/geodesic.pdf)、[LaTeX](notes/geodesic.tex)。
- 一般の二状態間の距離、分布の幅、位相の欠落：[追補PDF](docs/state_distance.pdf)、[LaTeX](notes/state_distance.tex)。
- 確率流、位相復元、長さの共役運動量：[追補PDF](docs/phase_space.pdf)、[LaTeX](notes/phase_space.tex)。
- [自動検証とPDF生成の実行履歴](https://github.com/ryos-physrockme/wasserstein-holography/actions)。

追補も独立して読めるように定義と規約を記載しています。以前の研究ノートは保持しています。ソース更新時にGitHub Actionsがテスト、数値再計算、四つのPDFのコンパイルを行い、`docs/`と`results/`を更新します。最新のビルドが未完了の場合、PDF・結果の更新はソースより遅れます。

## 確率流と共役運動量

同じchord基底で、隣接振幅の積 `c_n=psi_n^* psi_(n+1)` を測ります。確率流 `J_n=2 b_(n+1) Im(c_n)` は相対位相の正弦、エネルギーの各隣接項は余弦に依存します。長さ分布に確率流を加えると正負時間の状態を区別できますが、一般には位相の枝が残ります。

三基底からなる純粋状態の明示例では、全ての確率・確率流・総エネルギーが同じで、fidelityは1/4です。実Hamiltonianの `D K D=4B I-K`、`D_nn=(-1)^n` という関係により、この曖昧性は全時間で残ります。低温の熱的状態に限定した族での縮退を主張するものではありません。

純粋状態で非零確率の支持が連結なら、確率と隣接振幅の実部・虚部から全体位相を除いて状態を復元できます。混合状態や途中に零点がある場合には、この情報だけでは不十分です。

既存の重力差分方程式を正規直交化して、半古典Hamiltonian `h=2 Omega (1-sqrt(1-exp(-ell))*cos(p))` を照合します。`p_coh=arg(<A>)`、`A|n>=sqrt(1-q^n)|n-1>` は量子状態で測れる位相であり、大域的な運動量演算子の期待値とは区別します。`beta*Omega=10`、`0<=Omega*t<=10`で半古典軌道との差を確認し、有限鎖の端点補正を含む演算子恒等式と、倍の基底数での計算を検査します。

実装は `src/phase_space.py`、テストは `tests/test_phase_space.py`、結果は `results/phase_space/` です。確率と位相の正準形式は既知の量子力学の構造です。この照合を新しい電磁場や重力理論の導出とは扱いません。

## 一般の二状態間の距離

同じHamiltonian・基底・長さ演算子 `ell=epsilon*n` を使い、異なる温度・時間で準備した状態を比較します。長さの測定分布のWasserstein距離は、傾きの絶対値が1以下の長さの関数に限定した観測量による期待値差の上限に等しい。この双対表示を離散的な部分積分で導出し、最適な観測量も数値的に構成します。

同じ平均長の熱的状態では、距離は幅の違いを検出します。調べた `epsilon=0.05,0.02,0.01,0.005`、`beta_A*Omega=5`、`beta_B*Omega=10` ではWasserstein距離はほぼsqrt(epsilon)に比例し、Fisher–Rao距離は約0.067にとどまります。Gaussian公式は測定した幅による説明であり、幅の独立な理論予測や厳密な漸近定理ではありません。

同じ熱的状態から正負の時間で作る二状態は、実Hamiltonianのもとで長さ分布が厳密に同じですが、長さの変化率は逆符号です。Wasserstein距離も古典的Fisher–Rao距離も、測定で落とした相対位相を復元しません。既存の長さ演算子対応に基づく限定された状態の区別であり、一般の時空測地線や電磁場の導出とは区別します。

`src/state_distance.py`、`tests/test_state_distance.py`、`results/state_distance/` に実装・検査・分布CSV・最適な観測量・図・誤差を保存します。熱的準備後の基底は作り直さず、実時間確率は再規格化しません。打切りを倍にした比較、独立の輸送線形計画法、密行列指数関数、確率流・fidelity保存の検査を含みます。

## 有限温度での比較

入力する隣接係数は `b_n=a sqrt((1-q^n)/(1-q))` です。`epsilon=-log(q)`、`B=a/sqrt(1-q)`、`Omega=epsilon B` と定義します。逆温度betaの状態は、ゼロchord状態から `exp(-beta H/2)` で準備してから実時間発展させます。基底は準備前のゼロchord基準で固定し、熱的準備後にLanczos計算をやり直しません。

無次元長 `ell_beta(t)=epsilon sum_n n P_beta,n(t)` の増加分を、AdS2（二次元anti-de Sitter時空）の二つの境界を結ぶ測地線長と比較します。時間は左右の未来向き境界時間の和 `t=t_L+t_R` です。

先頭次数の熱的鞍点は `beta Omega sin(u)=pi-2u`、`v=1-2u/pi` で決まり、長さの増加は `2 log cosh(pi v t/beta)` です。vは時間データへのフィットではありません。微視的温度Tと、有効計量のHawking温度vTを区別します。低温でvが1に近づいたとき、同じ微視的温度のJackiw–Teitelboim重力の式に近づきます。小さいepsilonの極限と低温極限は別です。

これはHeller–Papalini–Schuhmannの既存の長さ対応を再現する検証です。固定電荷ごとに同じ形式のHamiltonianを代入する計算は行いますが、全電荷sectorを一つの荷電重力作用に組み込んだり、電磁場を導出したりしたわけではありません。

## 以前の距離幾何の検証

初期分布delta_0と整数間のコスト|n-m|を使うと、`W_1(P(t),delta_0)=sum_n n P_n(t)` は恒等式です。しかし任意の二時刻間の距離は平均の差より大きいことがあります。累積確率 `F_k=sum_{n<=k}P_n` と確率流 `J_k=2 b_(k+1) chi_k chi_(k+1)` は `dF_k/dt=-J_k` を満たし、分布の順序条件が直線距離への等長埋込みを決めます。

q=1をa固定で取るとPoisson分布となり、時刻順の直線性は厳密です。有限qでは厳密性が破れる例があり、距離行列の固有値解析と全時刻対の比較を実装しています。入力係数からの短時間展開、log-cosh近似のt^6でのずれ、Fisher計量とBerry曲率も元のノートで検証します。

## 再実行

Python 3.12とLuaLaTeXを使い、リポジトリ最上位で実行します。

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python src/krylov.py --output results
OPENBLAS_NUM_THREADS=1 python src/geodesic.py --output results/geodesic
OPENBLAS_NUM_THREADS=1 python src/state_distance.py --output results/state_distance
OPENBLAS_NUM_THREADS=1 python src/phase_space.py --output results/phase_space
for topic in note geodesic state_distance phase_space; do
    latexmk -lualatex -interaction=nonstopmode -halt-on-error -outdir=build "notes/${topic}.tex"
done
```

UbuntuではPDF用に `latexmk texlive-luatex texlive-lang-japanese texlive-latex-extra` を導入します。フォントは同梱しません。

`src/krylov.py`と`tests/test_krylov.py`はゼロchord初期状態と距離幾何、`src/geodesic.py`と`tests/test_geodesic.py`は熱的準備と測地線の比較を担当します。後者は独立の密行列指数関数、境界の時計、埋込み空間の内積、鎖の打切り、確率総和・エネルギー保存を検査します。`results/geodesic/summary.json`に実行環境と誤差を、`thermal_scan.csv`に数値表を保存します。

## 関連原典

Stefan Förste, Yannic Kruse, Saurabh Natu, *Grand Canonical vs Canonical Krylov Complexity in Double-Scaled Complex SYK Model*, [arXiv:2512.07715v2](https://arxiv.org/abs/2512.07715v2)。入力は式(3.33),(3.34)。

Michał P. Heller, Jacopo Papalini, Tim Schuhmann, *Krylov spread complexity as holographic complexity beyond JT gravity*, [arXiv:2412.17785v2](https://arxiv.org/abs/2412.17785v2)。有限温度、基準状態、長さ演算子、有効計量、正準形式、左右固有関数と正規直交化の対応。

Koji Hashimoto, Norihiro Tanahashi, *Holography and Optimal Transport: Emergent Wasserstein Spacetime in Harmonic Oscillator, SYK and Krylov Complexity*, [arXiv:2604.17649](https://arxiv.org/abs/2604.17649)。

Eliezer Rabinovici, Adrián Sánchez-Garrido, Ruth Shir, Julian Sonner, *A bulk manifestation of Krylov complexity*, [arXiv:2305.04355v2](https://arxiv.org/abs/2305.04355v2)。

M. Ambrosini, E. Rabinovici, A. Sánchez-Garrido, R. Shir, J. Sonner, *Operator K-complexity in DSSYK: Krylov complexity equals bulk length*, [arXiv:2412.15318v2](https://arxiv.org/abs/2412.15318v2)。

Elena Gubankova, Subir Sachdev, Grigory Tarnopolsky, *Scaling limits of complex Sachdev-Ye-Kitaev models and holographic geometry*, [arXiv:2512.05294v2](https://arxiv.org/abs/2512.05294v2)。このGreen関数からの再構成との完全な電荷・結合定数の照合は未検証です。
