# Wasserstein holography: research notes and verification code

確率分布の輸送距離、情報幾何、量子状態の幾何が、ホログラフィーにおいて何を決めるかを検討する作業用リポジトリです。

## 研究ノート

- 日本語PDF: [docs/note.pdf](docs/note.pdf)
- LaTeX: [notes/note.tex](notes/note.tex)
- 自動計算・PDF生成: [GitHub Actions](https://github.com/ryos-physrockme/wasserstein-holography/actions)

PDFと `results/` は、ソース更新時にGitHub Actionsで生成・保存します。初回のビルドが終わるまではPDFのリンク先は存在しません。各実行の成果物にもPDF・図・CSV・検証結果を保存します。

## 現在検証している問題

固定電荷complex Sachdev–Ye–Kitaev模型の有効三重対角Hamiltonianについて、Krylov基底上の確率分布を計算します。入力は Stefan Förste, Yannic Kruse, Saurabh Natu, *Grand Canonical vs Canonical Krylov Complexity in Double-Scaled Complex SYK Model*, [arXiv:2512.07715v2](https://arxiv.org/abs/2512.07715v2), Eqs. (3.33), (3.34) です。

`a=b_1`、`0<q<1` として、実装する係数は

```text
b_n = a sqrt((1-q^n)/(1-q)), n >= 1.
```

時間発展の初期状態は基底番号0です。測定結果nの確率をP_n(t)、その平均をC(t)とします。

確認済みの内容は以下です。

1. 輸送コストを |n-m| とすれば、初期分布delta_0からのW_1はCに厳密に等しい。ただし、二時刻間の距離は一般に平均の差より大きい。
2. 累積確率 `F_k(t)=sum_{n<=k} P_n(t)` とKrylov番号方向の確率流 `J_k=2 b_{k+1} chi_k chi_{k+1}` は `dF_k/dt=-J_k` を満たす。全リンクの流れが外向きなら、時刻順の分布は一次確率優越の順序にあり、任意の二時刻で `W_1=|C(t_2)-C(t_1)|` となる。
3. `q=1` 極限では `b_n=a sqrt(n)` となり、`P_n(t)` は平均 `(a t)^2` のPoisson分布である。この極限では距離行列は厳密な1次元直線距離になる。有限qでは内向き確率流が生じて厳密性は破れるが、qが1に近い有限時間窓ではずれは非常に小さい。
4. 入力した漸化式から `C(t)=a^2 t^2+O(t^4)` が従う。参照論文v2のEqs. (3.39), (3.43)に印刷された短時間係数とは整合しない。微視的模型から入力式までの全導出や、著者の図の生成コードを検証したという意味ではない。
5. `2/(1-q) log cosh(a sqrt(1-q) t)` はt^2,t^4係数を再現するが、t^6で厳密解と異なる。小さい変形では良い近似になる。平均の加速度について `C''=2 a^2 <q^n>` が成り立ち、Jensenの不等式からlog-cosh型の下界が得られる。
6. 同じ分布族の時間方向のFisher計量は、確率の零点を連続的に扱えば `g_tt=4a^2`。Krylov基底番号を生成子とする位相方向のBerry曲率も計算するが、その番号は保存電荷ではなく、曲率をbulk電磁場とは同一視しない。

これは再現計算と近似の検証です。log-cosh型の半古典的Krylov成長は既知であり、関数形だけを新規性として扱いません。Green関数からのbulk再構成との定量的な対応は未検証です。ノートに定義、導出、限定条件、原典を記載しています。

## 再実行

Python 3.12とLuaLaTeXを使用します。リポジトリの最上位で実行してください。

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python src/krylov.py --output results
latexmk -lualatex -interaction=nonstopmode -halt-on-error -outdir=build notes/note.tex
```

UbuntuではPDF用に `latexmk texlive-luatex texlive-lang-japanese texlive-latex-extra` を導入します。フォントファイルはリポジトリに同梱しません。

## ファイル

`src/krylov.py` は入力係数、疎行列指数関数による時間発展、輸送距離・計量の検査、図と数値表の生成を担当します。`tests/test_krylov.py` には記号計算によるt^6までの検算、別の行列指数関数実装との比較、独立の輸送距離実装との比較などを含みます。

`results/summary.json` は実行環境、打切り依存性、確率総和、恒等式の残差、近似の誤差に加えて、全時刻対のWasserstein距離行列、一次元直線距離からの偏差、classical multidimensional scalingの固有値、内向き確率流の診断を保存します。`results/geometry_scan.csv` はqごとの距離偏差を、`results/curves.csv` は `x=a sqrt(1-q) t` に対する `(1-q)C/2` を保存します。

## 関連原典

Koji Hashimoto, Norihiro Tanahashi, *Holography and Optimal Transport: Emergent Wasserstein Spacetime in Harmonic Oscillator, SYK and Krylov Complexity*, [arXiv:2604.17649](https://arxiv.org/abs/2604.17649).

M. Ambrosini, E. Rabinovici, A. Sánchez-Garrido, R. Shir, J. Sonner, *Operator K-complexity in DSSYK: Krylov complexity equals bulk length*, [arXiv:2412.15318v2](https://arxiv.org/abs/2412.15318v2).

Elena Gubankova, Subir Sachdev, Grigory Tarnopolsky, *Scaling limits of complex Sachdev-Ye-Kitaev models and holographic geometry*, [arXiv:2512.05294v2](https://arxiv.org/abs/2512.05294v2).
