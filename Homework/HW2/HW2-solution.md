# HW2 Solution

---

## 1. 马尔可夫决策过程

### 1 (a)

| $i$  | $V^{(i)}(-2)$ | $V^{(i)}(-1)$ | $V^{(i)}(0)$ | $V^{(i)}(1)$ | $V^{(i)}(2)$ |
| :--: | :-----------: | :-----------: | :----------: | :----------: | :----------: |
|  0   |       0       |       0       |      0       |      0       |      0       |
|  1   |       0       |      6.7      |      -1      |     5.3      |      0       |
|  2   |       0       |     7.17      |     4.65     |     5.86     |      0       |

### 1 (b)

$$
\mu(s)=
\begin{cases}
a_1, & s=-1 \\
a_1, & s=0 \\
a_2, & s=1 & 
\end{cases}
$$

---

## 2. Q-Learning

### 2 (a)

$$
\begin{align*}
Q(s,a) &= \mathbb{E}[G_t|s_t=s, a_t=a] \\
&= \mathbb{E}[\sum_{k=t}^{+\infty}\gamma^{k-t}\mathcal{R}_k|s_t=s, a_t=a] \\
 & =E\left[\mathcal{R}_{t}+\gamma\sum_{k=t+1}^{\infty}\gamma^{k-t-1}\mathcal{R}_{k}|s_{t}=s,a_{t}=a\right] \\
 & =\sum_{s^{\prime}}P(s^{\prime}|s,a)\left[\mathcal{R}(s,a,s^{\prime})+\gamma E\left[\sum_{k=t+1}^{\infty}\gamma^{k-t-1}\mathcal{R}_{k}|s_{t+1}=s^{\prime}\right]\right] \\
 & =\sum_{s^{\prime}} P(s^{\prime}|s,a)\left[\mathcal{R}(s,a,s^{\prime})+\gamma V(s^{\prime})\right]
\end{align*}
$$

### 2 (b)

给定一组 $(s,a,r,s^\prime)$，学习率 $\eta$ 和衰减因子 $\gamma$，Monte-Carlo Q-Learning 更新公式（详见课件 `6-强化学习.pdf` 第 30 页）：
$$
\hat{Q}_{\text{opt}}(s,a)\leftarrow(1-\eta)\hat{Q}_{\text{opt}}(s,a)+\eta(r+\gamma\underset{a^{\prime}\in\text{Actions}(s^\prime)}{\max}\hat{Q}_{\text{opt}}(s',a'))
$$
我们默认所有的 $\hat{Q}_{\text{opt}}(s,a)$ 为 0。

$t=1$ 时，四元组 $(0, a_1, 2, 1)$，更新如下：
$$
\begin{align*}
Q(0, a_1)&=(1-\eta)Q(0,a_1)+\eta(1+\gamma \cdot 0) \\
&=\eta
\end{align*}
$$
$t=2$ 时，四元组 $(1, a_1, 3, 0)$，更新如下：
$$
\begin{align*}
Q(1,a_1)=&\eta\cdot(3+\gamma\cdot Q(0, a_1))\\
=&\eta\cdot(3+\gamma\cdot\eta)
\end{align*}
$$
$t=3$ 时，四元组 $(0, a_2, -1, 1)$，更新如下：
$$
\begin{align*}
Q(0,a_2)&=\eta\cdot(-1+\gamma\cdot Q(1, a_1))\\
&=\eta \cdot (\gamma^2\eta^2+3\gamma\eta-1)
\end{align*}
$$

这道题实际上只需要计算 $t=1,2,3$ 即可，作业文档的表达 $1\leq t\leq 4$ 的表述存在问题。**这道题如有计算错误，不予扣分。**

### 2(c)

概述参考资料的内容，说明在算法迭代过程中，估计值与真实值之间的差距逐渐减小即可。


---

## 3. Gobang Programming

参考代码 `solution.py` 将上传至 bb 系统。

在默认 setting 下，3x3 的 Gobang 胜率一般能够达到 90%+， 4x4 的胜率一般在 50% 左右。4x4 胜率下降的原因是可能的状态空间远多于 3x3，在 epoch 不变的情况下，学习到的 Q table 与真实值存在一定差距（学习效果不佳）。

---

## 4. Deeper Understanding

### 4.1 (a)

$$
\begin{aligned}
\|\mathcal{T}v_1-\mathcal{T}v_2\|_\infty & =\gamma\Vert\max_{a\in A}\sum_{s^{\prime}\in S}P_{sa}(s^{\prime})\left[v_1(s^{\prime})-v_2(s^{\prime})\right]\Vert_{\infty} \\
 & =\gamma\max_{s^{\prime}\in S}|\max_{a\in A}\sum_{s^{\prime}\in S}P_{sa}(s^{\prime})\left[v_1(s^{\prime})-v_2(s^{\prime})\right]| \\
 & \leq\gamma\|v_1-v_2\|_\infty
\end{aligned}
$$

### 4.1 (b)

假设 $V_1$ 与 $V_2$ 均为不动点，i.e., $B(V_1) = V_1$, $B(V_2) = V_2$.
$$
\|V_1 - V_2\|_\infty = \|B(V_1) - B(V_2)\|_\infty \leq \gamma \|V_1 - V_2\|_\infty \\
\|V_1 - V_2\|_\infty = 0 \\
V_1 = V_2
$$

### 4.2(a)

$$
\begin{align*}
\mathbb{E}_{x\sim p}[f(x)]
&=\int_x p(x)f(x)dx\\
&=\int_x q(x)\frac{p(x)}{q(x)}f(x)dx\\
&=\mathbb{E}_{x\sim q}[f(x)\frac{p(x)}{q(x)}]
\end{align*}
$$

### 4.2(b)

$$
\begin{align*}
&\text{Var}_{x\sim p}[f(x)]\\
=& \mathbb{E}_{x\sim p}[f(x)^2]-(\mathbb{E}_{x\sim p}[f(x)])^2\\
\\
&\text{Var}_{x\sim q}[f(x)\frac{p(x)}{q(x)}]\\
=& \mathbb{E}_{x\sim q}[f(x)^2\frac{p(x)^2}{q(x)^2}]-(\mathbb{E}_{x\sim q}[f(x)\frac{p(x)}{q(x)}])^2 \\
=&\mathbb{E}_{x\sim p}[f(x)^2\frac{p(x)}{q(x)}]-(\mathbb{E}_{x\sim p}[f(x)])^2
\end{align*}
$$


$$
\begin{align*}
&\text{Var}_{x\sim p}[f(x)]-\text{Var}_{x\sim q}[f(x)\frac{p(x)}{q(x)}]\\
=& \mathbb{E}_{x\sim p}[f(x)^2]-\mathbb{E}_{x\sim p}[f(x)^2\frac{p(x)}{q(x)}]
\end{align*}
$$
