# Information geometry and holography: research notes and verification

確率分布の輸送距離、Fisher計量、双対接続、量子状態の幾何がホログラフィーの何に対応するかを検討するリポジトリです。入力、導出、数値的証拠、既存の対応の再現、未検証の仮説を区別します。

## 現在の方針（2026年9月16日）

**[研究方針の再評価PDF](docs/reassessment.pdf) / [LaTeX](notes/reassessment.tex)** に、文献調査、従来の計算の位置づけ、新規性・進歩性・物理的意義の判定条件をまとめました。FisherによるAdS時空構成、量子相対エントロピーと正準エネルギー、Berry/Uhlmann/modular接続とバルクのシンプレクティック形式・曲率、charged entanglement first lawからのゲージ場の線形方程式を比較します。

Wassersteinを唯一の幾何とは前提しません。従来のchord計算は再現・診断の基盤として保存し、荷電chord挿入の実装を自動的に次の主題とする方針は保留します。今後は、同じ物理的なsource状態族について、各情報量が何を識別し、どのバルク量を再構成するかを先に固定します。

次の候補は、保存電流のsourceで準備した状態とMaxwell場の変分を、ゲージ条件・境界電束条件・正則化までそろえて比較することです。既知のBerry--symplectic対応を再現するだけでは新規性としません。局在荷電プローブの閉経路位相、dressing依存性、有限分解能の補正などに独立の予測が残るかを判定します。自由Maxwell場のシンプレクティック形式は背景場自身に依存しないので、それを局所電場の値と取り違えません。

双対接続の候補では、source・期待値の双対性と三次応答から始めます。三点相関をAmari--Chentsovテンソルと呼び直すだけでは、既存の場の理論の情報幾何を超えません。量子のdivergenceと正則化を固定し、ゲージ不変でsourceの座標に依存しない新しい関係が残るかを検査します。

## 解析的な基準計算

`src/metric_benchmarks.py` と `tests/test_metric_benchmarks.py` は、規格化した四次元instanton密度

```text
p(x;a,rho) = 6/pi^2 * rho^4 / (|x-a|^2+rho^2)^4
```

のFisher計量と三次テンソルを記号積分で検算します。

```text
Fisher: ds^2 = (16/5) (|da|^2+drho^2)/rho^2
W2:     distance^2 = |a-b|^2+2(rho-sigma)^2  (Euclidean ground cost)
C_ijk = 0: all alpha connections coincide with Levi-Civita, whose curvature is nonzero.
```

Fisher計量と三次テンソルの消失はBlau--Narain--Thompson、hep-th/0108122の既知の結果です。W2の式もlocation--scale族の標準的な輸送の計算です。正規分布族のW1微小ノルムが一般に二次形式ではないことも検算します。これらを新しいホログラフィック対応とは扱いません。

## 研究ノート一覧

各ノートは独立に読めるよう、模型・記号・近似・適用範囲を定義しています。

- 現行方針と文献調査：[PDF](docs/reassessment.pdf)、[LaTeX](notes/reassessment.tex)
- 距離行列・確率流・Fisher計量・Berry曲率：[PDF](docs/note.pdf)、[LaTeX](notes/note.tex)
- 有限温度のchord分布と測地線長：[PDF](docs/geodesic.pdf)、[LaTeX](notes/geodesic.tex)
- 二状態間の距離・幅・位相の欠落：[PDF](docs/state_distance.pdf)、[LaTeX](notes/state_distance.tex)
- 確率流・位相復元・長さの共役運動量：[PDF](docs/phase_space.pdf)、[LaTeX](notes/phase_space.tex)
- bilocal応答と隣接コヒーレンス：[PDF](docs/boundary_probe.pdf)、[LaTeX](notes/boundary_probe.tex)
- 荷電probeの相関関数の偶・奇分解：[PDF](docs/charged_probe.pdf)、[LaTeX](notes/charged_probe.tex)

以前のノートにある「次の課題」の提案は当時の位置づけです。現在の優先順位はreassessmentを参照してください。既存の導出・コード・結果は削除していません。

## 従来の検証結果の範囲

初期点delta_0までのW1が平均Krylov番号に等しいことは定義から従います。任意の二時刻間の距離が平均差に等しいかは別問題であり、累積確率の順序と確率流で検査しました。同じ平均長でも分布の幅が違えばW1は非零です。固定基底の確率だけでは相対位相が失われますが、これは一般のHusimi Qが情報を失うという主張ではありません。

有限温度の平均chord数・長さ・半古典運動量の比較はHeller--Papalini--Schuhmann等の既存の対応を入力とする再現です。基準状態と温度の規約を固定し、熱的準備後に基底を作り直していません。小さい変形と低温のJackiw--Teitelboim領域は別の極限です。

中性bilocalの時間微分と逆温度微分から、隣接コヒーレンスの生成関数を得る恒等式を検証しました。無限精度の関数からの一意性と、有限個のデータからの安定な復元は区別します。荷電二点関数の偶・奇分解は既存の固定電荷・grand-canonical表示の照合であり、Wassersteinの接続からMaxwell場を導出したものではありません。

## 再実行と自動コンパイル

Python 3.12とLuaLaTeXを使用します。GitHub Actionsは全テスト、各計算、全ノートのコンパイルを行い、`docs/`と`results/`を保存します。実行途中にはPDFがソースより遅れる場合があります。

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python src/krylov.py --output results
for topic in geodesic state_distance phase_space boundary_probe charged_probe metric_benchmarks; do
    OPENBLAS_NUM_THREADS=1 python "src/${topic}.py" --output "results/${topic}"
done
for topic in note geodesic state_distance phase_space boundary_probe charged_probe reassessment; do
    latexmk -lualatex -interaction=nonstopmode -halt-on-error -outdir=build "notes/${topic}.tex"
done
```

Ubuntuでは`latexmk texlive-luatex texlive-lang-japanese texlive-latex-extra`を使用します。フォントファイルは配布しません。[自動実行履歴](https://github.com/ryos-physrockme/wasserstein-holography/actions)。

## 方針に直接関係する原典

- M. Blau, K. S. Narain, G. Thompson, *Instantons, the Information Metric, and the AdS/CFT Correspondence*, [hep-th/0108122](https://arxiv.org/abs/hep-th/0108122)
- J. Erdmenger, K. T. Grosvenor, R. Jefferson, *Information geometry in quantum field theory: lessons from simple examples*, [2001.02683](https://arxiv.org/abs/2001.02683)
- N. Lashkari, M. Van Raamsdonk, *Canonical Energy is Quantum Fisher Information*, [1508.00897](https://arxiv.org/abs/1508.00897)
- S. Floerchinger, *Information geometry of Euclidean quantum fields*, [2303.04081](https://arxiv.org/abs/2303.04081)
- A. Belin, A. Lewkowycz, G. Sarosi, *The boundary dual of the bulk symplectic form*, [1806.10144](https://arxiv.org/abs/1806.10144)
- J. Kirklin, *The Holographic Dual of the Entanglement Wedge Symplectic Form*, [1910.00457](https://arxiv.org/abs/1910.00457)
- B. Czech et al., *A Modular Sewing Kit for Entanglement Wedges*, [1903.04493](https://arxiv.org/abs/1903.04493)
- B. Czech et al., *Changing states in holography: From modular Berry curvature to the bulk symplectic form*, [2305.16384](https://arxiv.org/abs/2305.16384)
- K. Hasegawa, Y. Tanii, *Linearized Field Equations of Gauge Fields from the Entanglement First Law*, [1905.10084](https://arxiv.org/abs/1905.10084)

Wasserstein、chord、荷電SYKの原典および他の情報幾何的アプローチを含む文献一覧は、各ノートとreassessment末尾を参照してください。
