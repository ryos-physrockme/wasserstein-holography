# Wasserstein holography: research notes and verification code

確率分布の輸送距離、情報幾何、量子状態の幾何が、ホログラフィーにおいて何を決めるかを検討するリポジトリです。入力式、厳密な恒等式、近似、数値結果、未検証の重力的解釈を区別します。

## 研究ノート

- 距離行列・確率流・Fisher計量・Berry曲率：[PDF](docs/note.pdf)、[LaTeX](notes/note.tex)
- 有限温度のchord分布と二境界間の測地線長：[PDF](docs/geodesic.pdf)、[LaTeX](notes/geodesic.tex)
- 一般の二状態間の距離、分布幅、位相の欠落：[PDF](docs/state_distance.pdf)、[LaTeX](notes/state_distance.tex)
- 確率流、位相復元、長さの共役運動量：[PDF](docs/phase_space.pdf)、[LaTeX](notes/phase_space.tex)
- bilocal probeの微分から隣接コヒーレンスを復元：[PDF](docs/boundary_probe.pdf)、[LaTeX](notes/boundary_probe.tex)
- [GitHub Actions](https://github.com/ryos-physrockme/wasserstein-holography/actions)

各追補はチャット履歴なしで読めるように、模型・記号・近似・適用範囲を定義しています。ソース更新時にGitHub Actionsが全テスト、数値再計算、各PDFのコンパイルを実行し、`docs/`と`results/`を更新します。

## bilocal probeと隣接コヒーレンス

長さ演算子を `ell=epsilon*N`、bilocal probeを `Q_Delta=exp(-Delta*ell)=z^N`, `z=exp(-epsilon*Delta)` とします。熱的準備状態における期待値

```text
B_Delta(t,beta) = <Q_Delta> = sum_n P_n z^n
```

を、長さ測定分布の確率生成関数として使います。neutral double-scaled SYK / sine-dilaton gravityでは、このoperator familyに既存のmassive-probe bilocal辞書があります。

隣接振幅積を `c_n=psi_n^* psi_(n+1)` とし、確率流 `J_n` とhopping energy `h_n` を

```text
J_n =  2 B sqrt(1-q^(n+1)) Im(c_n)
h_n = -2 B sqrt(1-q^(n+1)) Re(c_n)
```

と定義します。それぞれの生成関数は、同じbilocal期待値の外部パラメータ微分だけで厳密に

```text
J_Delta = (partial_t B_Delta)/(z-1)
H_Delta = 2[(E_beta-2B) B_Delta - partial_beta B_Delta]/(1+z)
```

と書けます。従って複素生成関数

```text
C_Delta = sum_n sqrt(1-q^(n+1)) c_n z^n
```

は

```text
C_Delta = [partial_beta B_Delta + (2B-E_beta)B_Delta]/[B(1+z)]
          + i [partial_t B_Delta]/[2B(z-1)]
```

で得られます。時間微分はcommutator、逆温度微分はconnected anticommutatorを測ります。全Delta依存性を無限精度で知れば、解析関数の一意性から隣接コヒーレンスの係数は原理上決まりますが、有限点からの逆問題はVandermonde型で急速に悪条件になります。

実装は `src/boundary_probe.py`、テストは `tests/test_boundary_probe.py`、結果は `results/boundary_probe/` です。恒等式をcommutator/anticommutatorから直接検査すると約1e-12以下で一致し、`t`・`beta`を実際にずらした有限差分でも独立に確認します。

この結果はchord Hamiltonian上の代数恒等式です。neutral DSSYKのbilocal辞書と組み合わせるとboundary側の意味を持ちますが、fixed-charge complex SYKで同じoperatorがどのcharged matter insertionに対応するかはまだ確立していません。復元した位相をbulk electromagnetic fieldと同一視しません。

## 確率流と共役運動量

同じchord基底で隣接振幅積 `c_n` を考えると、確率流は相対位相の正弦、隣接energyは余弦に依存します。確率・全確率流・総エネルギーが同一でも異なる純粋状態が存在する明示例を構成しています。純粋状態で非零確率の支持が連結なら、確率と隣接振幅の実部・虚部から全体位相を除いて状態を復元できます。混合状態や零点がある場合には追加情報が必要です。

半古典Hamiltonian `h=2 Omega [1-sqrt(1-exp(-ell))*cos(p)]` と照合し、`p_coh=arg(<A>)` が小さいepsilonで古典的な共役運動量へ近づくことを数値確認します。`p_coh`は複素期待値の偏角であり、自己共役な大域運動量演算子の期待値と主張しません。

## 一般の二状態間の距離

同じHamiltonian・基底・長さ演算子を使う二状態について、長さ測定分布の1-Wasserstein距離は、1-Lipschitzな長さの関数に限定した期待値差の上限に等しいことを直接導出しています。同じ平均長でも分布幅が異なれば距離は非零です。比較した熱的状態ではWasserstein距離はほぼ`sqrt(epsilon)`に比例し、Fisher–Rao距離は有限値に残ります。

一方、長さ分布が厳密に同じでも相対位相が異なる状態はWasserstein距離でも古典的Fisher–Rao距離でも区別できません。正負時間の状態を使って、その後の時間発展が異なることを検証しています。

## 有限温度での長さ比較

逆温度`beta`の状態をゼロchord状態から`exp(-beta H/2)`で準備し、準備前の固定基底で実時間発展させます。`epsilon=-log(q)`, `B=a/sqrt(1-q)`, `Omega=epsilon B`と定義し、無次元長`ell_beta(t)=epsilon sum_n n P_beta,n(t)`を重力側の二境界間の長さと比較します。

先頭次数の熱的鞍点は`beta Omega sin(u)=pi-2u`, `v=1-2u/pi`で決まり、長さ増加は`2 log cosh(pi v t/beta)`です。`v`は時間データへのフィットではありません。小さい`epsilon`の極限と低温のJackiw–Teitelboim領域は別の極限であることを数値的に区別します。

## 距離行列の一次元性

初期分布`delta_0`とコスト`|n-m|`では`W_1(P(t),delta_0)=sum_n nP_n(t)`は恒等式です。ただし一般の二時刻間では平均の差より大きくなり得ます。累積確率`F_k`と確率流`J_k`は`dF_k/dt=-J_k`を満たし、時刻順の一次確率優越が厳密な直線距離への等長埋込み条件になります。`q=1`を`a`固定で取るとPoisson分布となり直線性は厳密、有限`q`では内向き流によって厳密性が破れます。

## 再実行

Python 3.12とLuaLaTeXを使用します。

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python src/krylov.py --output results
OPENBLAS_NUM_THREADS=1 python src/geodesic.py --output results/geodesic
OPENBLAS_NUM_THREADS=1 python src/state_distance.py --output results/state_distance
OPENBLAS_NUM_THREADS=1 python src/phase_space.py --output results/phase_space
OPENBLAS_NUM_THREADS=1 python src/boundary_probe.py --output results/boundary_probe
for topic in note geodesic state_distance phase_space boundary_probe; do
    latexmk -lualatex -interaction=nonstopmode -halt-on-error -outdir=build "notes/${topic}.tex"
done
```

UbuntuではPDF用に`latexmk texlive-luatex texlive-lang-japanese texlive-latex-extra`を使用します。フォントファイルはリポジトリに含めません。

## 主な原典

- Stefan Förste, Yannic Kruse, Saurabh Natu, *Grand Canonical vs Canonical Krylov Complexity in Double-Scaled Complex SYK Model*, [arXiv:2512.07715v2](https://arxiv.org/abs/2512.07715v2)
- Michał P. Heller, Jacopo Papalini, Tim Schuhmann, *Krylov spread complexity as holographic complexity beyond JT gravity*, [arXiv:2412.17785v2](https://arxiv.org/abs/2412.17785v2)
- Koji Hashimoto, Norihiro Tanahashi, *Holography and Optimal Transport: Emergent Wasserstein Spacetime in Harmonic Oscillator, SYK and Krylov Complexity*, [arXiv:2604.17649](https://arxiv.org/abs/2604.17649)
- Eliezer Rabinovici, Adrián Sánchez-Garrido, Ruth Shir, Julian Sonner, *A bulk manifestation of Krylov complexity*, [arXiv:2305.04355v2](https://arxiv.org/abs/2305.04355v2)
- M. Ambrosini et al., *Operator K-complexity in DSSYK: Krylov complexity equals bulk length*, [arXiv:2412.15318v2](https://arxiv.org/abs/2412.15318v2)
- Elena Gubankova, Subir Sachdev, Grigory Tarnopolsky, *Scaling limits of complex Sachdev–Ye–Kitaev models and holographic geometry*, [arXiv:2512.05294](https://arxiv.org/abs/2512.05294)
- E. Alfinito, M. Beccaria, *Krylov Correlators in sl(2,R) Models: Exact Results and Holographic Complexity*, [arXiv:2605.17550v4](https://arxiv.org/abs/2605.17550v4)
- D. Chatzis et al., *Holographic Spread Complexity at Fixed Charge: Routhians, Branes and Strings*, [arXiv:2608.23709](https://arxiv.org/abs/2608.23709)
