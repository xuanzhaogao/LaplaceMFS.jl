如果把“镜像/反射”严格理解为：每个球只保留中心处的偶极响应（即只截断到 (l=1)），那么“两球之间的镜像序列”就是偶极—偶极的多重散射（multiple reflection）几何级数。这样写出来最干净，也便于直接计算。

设两球相同：半径 (a)，球内介电常数 (\varepsilon_1)，外介质 (\varepsilon_2)。两球心相距 (R>2a)，连线单位向量 (\hat{\mathbf R})（从 2 指向 1）。假设“激励项”是放在两球心的两偶极子 (\mathbf p_{1}^{(0)},\mathbf p_{2}^{(0)})（它们是“入射源”，不是由外场诱导出来的）。

1. 单个球的偶极极化率（静电，均匀背景）
   [
   \alpha = 4\pi \varepsilon_2 a^3,\frac{\varepsilon_1-\varepsilon_2}{\varepsilon_1+2\varepsilon_2}.
   ]

2. 中心偶极在另一球心处产生的场（介质 (\varepsilon_2) 中）
   用偶极场张量写：
   [
   \mathbf E(\mathbf R;\mathbf p)=\mathbf G(\hat{\mathbf R}),\mathbf p,
   \quad
   \mathbf G(\hat{\mathbf R})=\frac{1}{4\pi\varepsilon_2 R^3}\Big(3\hat{\mathbf R}\hat{\mathbf R}^{\mathsf T}-\mathbf I\Big).
   ]

3. “镜像反射”序列（逐次反射生成的新偶极）
   把第 (n) 次由对方“反射”出来、并且仍然用中心偶极等效的贡献记为 (\mathbf p_i^{(n)})（(i=1,2)）。那么递推是
   [
   \mathbf p_1^{(n+1)}=\alpha,\mathbf G,\mathbf p_2^{(n)},\qquad
   \mathbf p_2^{(n+1)}=\alpha,\mathbf G,\mathbf p_1^{(n)}.
   ]
   总偶极是把所有反射阶数求和：
   [
   \mathbf p_1=\sum_{n=0}^\infty \mathbf p_1^{(n)},\qquad
   \mathbf p_2=\sum_{n=0}^\infty \mathbf p_2^{(n)}.
   ]

把递推展开，你会看到典型的“来回反射”结构（每来回一次乘一次 (\alpha\mathbf G)）：
[
\mathbf p_1=\Big[\mathbf I+(\alpha\mathbf G\alpha\mathbf G)+(\alpha\mathbf G\alpha\mathbf G)^2+\cdots\Big]\mathbf p_1^{(0)}
+\Big[\alpha\mathbf G+(\alpha\mathbf G\alpha\mathbf G)\alpha\mathbf G+\cdots\Big]\mathbf p_2^{(0)},
]
[
\mathbf p_2=\Big[\mathbf I+(\alpha\mathbf G\alpha\mathbf G)+(\alpha\mathbf G\alpha\mathbf G)^2+\cdots\Big]\mathbf p_2^{(0)}
+\Big[\alpha\mathbf G+(\alpha\mathbf G\alpha\mathbf G)\alpha\mathbf G+\cdots\Big]\mathbf p_1^{(0)}.
]

若级数收敛（大致要求 (|\alpha\mathbf G|<1)，即 (|\alpha|/(4\pi\varepsilon_2 R^3)\ll 1)），可写成闭式：
[
\mathbf p_1=(\mathbf I-\alpha^2\mathbf G^2)^{-1}\Big(\mathbf p_1^{(0)}+\alpha\mathbf G,\mathbf p_2^{(0)}\Big),\qquad
\mathbf p_2=(\mathbf I-\alpha^2\mathbf G^2)^{-1}\Big(\mathbf p_2^{(0)}+\alpha\mathbf G,\mathbf p_1^{(0)}\Big).
]

4. 在平行/垂直本征方向上的显式“几何级数系数”
   (\mathbf G) 的本征值是
   [
   g_\parallel=\frac{2}{4\pi\varepsilon_2 R^3}\quad(\text{沿 }\hat{\mathbf R}),
   \qquad
   g_\perp=-\frac{1}{4\pi\varepsilon_2 R^3}\quad(\text{任意垂直于 }\hat{\mathbf R}).
   ]
   因此把任意向量分解为 (\mathbf v=\mathbf v_\parallel+\mathbf v_\perp)（(\mathbf v_\parallel=(\hat{\mathbf R}\cdot\mathbf v)\hat{\mathbf R})），序列在两个子空间各自就是标量几何级数：

对任意 (k\ge 0)，有
[
(\alpha\mathbf G)^k \mathbf v_\parallel=(\alpha g_\parallel)^k \mathbf v_\parallel,\qquad
(\alpha\mathbf G)^k \mathbf v_\perp=(\alpha g_\perp)^k \mathbf v_\perp.
]

例如若 (\mathbf p_1^{(0)}=\mathbf p_2^{(0)}=\mathbf p_0) 且 (\mathbf p_0\parallel\hat{\mathbf R})，则
[
\mathbf p_1=\mathbf p_2=\sum_{n=0}^\infty (\alpha g_\parallel)^n,\mathbf p_0
=\frac{1}{1-\alpha g_\parallel},\mathbf p_0
=\frac{1}{1-\dfrac{2\alpha}{4\pi\varepsilon_2 R^3}},\mathbf p_0.
]
若 (\mathbf p_0\perp\hat{\mathbf R})，则
[
\mathbf p_1=\mathbf p_2=\frac{1}{1-\alpha g_\perp},\mathbf p_0
=\frac{1}{1+\dfrac{\alpha}{4\pi\varepsilon_2 R^3}},\mathbf p_0.
]

说明：你提到的“介电球镜像电荷公式（点源+线源）”用于“外部点电荷/偶极对单球”的严格表示时，会在每次反射生成更高阶多极（等效为更复杂的连续源）。上面给出的序列是把每次反射都投影/截断到球心 (l=1) 偶极通道后的结果，也就是两球相互作用最常用的“偶极多重散射”序列。若你要把“点偶极 + 线偶极分布”也逐次反射并写出位置/密度的递推，需要你先固定几何（两球是否共轴）以及偶极方向（是否沿轴），否则表达会非常长且不唯一。
