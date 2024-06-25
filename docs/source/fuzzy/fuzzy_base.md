# Fuzzy Sets

## Fuzzy Concepts

&emsp;&emsp;[**定义**]  **模糊集**$A$以一个隶属函数$\mu_A(x):U\rightarrow [0,1]$为规范。

&emsp;&emsp;表示方式有以下3种：

&emsp;&emsp;- $A=\{(x,\mu_A(x))|x\in U\}$

&emsp;&emsp;- $A=\int_U \mu_A(x)/x$

&emsp;&emsp;- $A=\sum_U \mu_A(x)/x$

&emsp;&emsp;[**定义**] 

$$
\mathrm{supp}(A)=\{\mu_A(x)>0|x\in U\}
$$

&emsp;&emsp;[**定义**] 

$$
\mathrm{height}(A)=\max\limits_{x\in U} \mu_A(x)
$$


&emsp;&emsp;[**定义**] **交叉点**

$$
\mathrm{crossover}(A)=\{\mu_A(x)=0.5|x\in U\}
$$

&emsp;&emsp;[**定义**] $\alpha-cut$

$$
A_\alpha = \{\mu_A(x)\ge \alpha|x\in U\}
$$

&emsp;&emsp;[**性质**] 若$A$为凸集，则有

$$
\mu_A(\lambda x_1 + (1-\lambda)x_2)\ge \min (\mu_A(x_1),\mu_A(x_2))
$$

## Fuzzy Operations

&emsp;&emsp;[**相等运算**]

$$
A=B \Longleftrightarrow \mu_A(x)=\mu_B(x) |x \in U
$$

&emsp;&emsp;[**包含运算**]

$$
A\subseteq B \Longleftrightarrow \mu_A(x)\le\mu_B(x) |x \in U
$$

&emsp;&emsp;[**补集运算**]

$$
\bar{A} \Longleftrightarrow \mu_{\bar{A}}(x)=1-\mu_A(x) |x \in U
$$

&emsp;&emsp;[**并集运算**]为s-norm特例

$$
A\cup B \Longleftrightarrow \mu_{A\cup B}(x)=\max(\mu_A(x),\mu_B(x)) |x \in U
$$

&emsp;&emsp;[**交集运算**]为t-norm特例

$$
A\cap B \Longleftrightarrow \mu_{A\cap B}(x)=\min(\mu_A(x),\mu_B(x)) |x \in U
$$

### S-Norms

&emsp;&emsp;[**定义**]: 若二元运算$s:[0,1]\times [0,1]\rightarrow [0,1]$满足以下条件，则称其为**S-Norm**(例：max)：

&emsp;&emsp;1. $s(1,1)=1, s(0,a)=s(a,0)=a$

&emsp;&emsp;2. $s(a,b)=s(b,a)$

&emsp;&emsp;3. if $a\le a',b\le b'$, then $s(a,b)\le s(a',b')$

&emsp;&emsp;4. $s[s(a,b),c]=s[a,s(b,c)]$

&emsp;&emsp;[**定义**]: 若给定s-norm $s$以及模糊集$A,B$，则广义模糊并集为，

$$
\mu_{A\cup B}(x)=s(\mu_A(x),\mu_B(x))
$$

&emsp;&emsp;[**例**]: $\mu_{A \cup B}(x)=\max(\mu_A(x),\mu_B(x))$。

#### 常用S-Norms

- Dombi

$$
s_\lambda(a,b)=\frac{1}{1+\left[(1/a-1)^{-\lambda} +(1/b-1)^{-\lambda}\right]^{-1/\lambda}},\quad \lambda\in(0,\infty)
$$

```matlab
N=100
m=2
a=linspace(0.01,0.99,N)
b=linspace(0.01,0.99,N)
A=((1./a)-1).^(-m)
B=((1./b)-1).^(-m)
[A,B]=meshgrid(A,B)
s=1+(A+B).^(-1/m)
s=1./s
surf(a,b,s)
```

- Dubois

 $$
    s_\alpha (a,b)=\frac{a+b-ab-\min(a,b,1-\alpha)}{\max(1-a,1-b,\alpha)},\quad \alpha\in(0,1)
 $$

 - Yager

 $$
s_\omega(a,b)=\min(1,(a^\omega+b^\omega)^{1/\omega}),\quad \omega\in(0,\infty)
 $$

 - Drastic sum: (上界)

 $$
s_{ds}(a,b)=\left\{\begin{array}{ll}a,&  b=0,\\ b,& a=0,\\ 1,& \mathrm{otherwise.}\\ \end{array} \right.
 $$

  - Einstein sum:

 $$
s_{es}(a,b)=\frac{a+b}{1+ab}
 $$


  - Algebratic sum:

 $$
s_{as}(a,b)=a+b-ab.
 $$

 - Maximum: (下界)

 $$
s_m(a,b)=\max(a,b)
 $$

 &emsp;&emsp;[**定理**]: s-norm的上、下界如下，

 $$
\max(a,b)\le s(a,b)\le s_{ds}(a,b)
 $$

 &emsp;&emsp;[**定理**]: 

 $$
\lim_{\lambda\rightarrow \infty}s_\lambda(a,b)=\max{a,b},\quad \lim_{\lambda\rightarrow 0}s_\lambda(a,b)=s_{ds}(a,b)
 $$
