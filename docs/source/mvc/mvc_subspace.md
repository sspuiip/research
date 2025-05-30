# Multi-View Subspace Clustering



## Multi-View data

Multi-view data set are generated by multiple distinct feature sets.

For example, an image can be represented by the color, shapes, texture and so on. a document is described by several different languages.

## Subspace clustering

Subspace learning assumes that the single source data are drawn from multiple low-dimensional subspaces.

- typical methods:
    - iteration based methods
    - factorization based methods
    - statistical approaches
    - spectral clustering based approaches


### Standard subspace clustering

Given $n$ data points $X=\{\mathbf{x}_1,\cdots,\mathbf{x}_n\}\in\mathbb{R}^{d\times n}$, the **self-expression property** of the data set are used to represent itself as,

$$
X=XZ+E
$$

where $Z=\{\mathbf{z}_1,\cdots,\mathbf{z}_n\}\in \mathbb{R}^{n\times n}$ is the subspace representation matrix, and each $\mathbf{z}_i$ stands for the data point $\mathbf{x}_i$ in the subspace. $E\in\mathbb{R}^{d\times n}$ is the error matrix.

<table><tr><td bgcolor="#f0f0f0">

We know that each data point $\mathbf{x}_j$ can be mapped to a new space through a dictionary  $B=[\mathbf{b}_1,\cdots,\mathbf{b}_k]$, i.e., $\mathbf{x}_j=\sum_{i=1}w_{ji}\mathbf{b}_i$. Self-expression  is just replacing the dictionary with data matrix itself.


$$
\mathbf{x}_j=\sum_{i=1}^k W_{ji}\mathbf{b}_i, \quad i.e., \begin{bmatrix}x_{j1}\\x_{j2}\\\vdots\\ x_{jd} \end{bmatrix}=\begin{bmatrix}| &|&\cdots &|\\ \mathbf{b}_1 &\mathbf{b}_2&\cdots &\mathbf{b}_k\\| &|&\cdots &|\\\end{bmatrix}\begin{bmatrix}w_{j1}\\w_{j2}\\\vdots\\ w_{jk} \end{bmatrix}
$$

so we have the following equation,

$$
\mathbf{X}=\mathbf{BW}
$$

</td></tr><table>


In general, we can obtain the value of $\mathbf{Z}$ through solving the following question, i.e.,

$$
\min\limits_{\mathbf{Z}}\quad \lVert \mathbf{X}-\mathbf{XZ}\rVert_F^2
$$

Then normalize the symmetric matrix $\mathbf{S} $,

$$
\mathbf{S}=|\mathbf{Z}|+|\mathbf{Z}^\top |
$$

Once we have the symmetric matrix, we can use any clustering method for clustering such as spectral clustering. 

## Multi-view Clustering

### Naive Multi-view Subspace clustering

Naive Multi-view Subspace clustering (NMVSC) first obtains the subspace representation in each view, then, fuses all subspace representation using some rules.

The object function of view $v$:

$$
\min\limits_{\mathbf{Z}^{(v)}} f(\mathbf{Z}^{(v)})=\lVert \mathbf{X}^{(v)}-\mathbf{X}^{(v)}\mathbf{Z}^{(v)}\rVert_F^2+\alpha^{(v)}\Omega(\mathbf{Z}^{(v)})
$$

where,

$$
\Omega(\mathbf{Z}^{(v)})=\frac12\sum_{i=1}^n\sum_{j=1}^nw_{ij}^{(v)}\lVert \mathbf{z}_i^{(v)}-\mathbf{z}_j^{(v)}\rVert_2^2=\mathrm{tr}(\mathbf{Z}^{(v)}\mathbf{L}\mathbf{Z}^{(v)^\top})
$$

Therefore, the object function of multi-view clustering becomes the following form,

$$
\Omega(\mathbf{Z}^{(1)},...,\mathbf{Z}^{(V)})=\sum_{v=1}^V\lVert \mathbf{X}^{(v)}-\mathbf{X}^{(v)}\mathbf{Z}^{(v)}\rVert_F^2 + \sum_{v=1}^V\alpha^{(v)}\mathrm{tr}(\mathbf{Z}^{(v)}\mathbf{L}^{(v)}\mathbf{Z}^{(v)^\top})
$$

[Paper with code](https://paperswithcode.com/task/multi-view-subspace-clustering#papers-list)


# Multi-view low-rank sparse subspace clustering

Some notations and abbreviations.

| Notation    | Definition      | 
|----|-----|
|  $N$   | Number of data points     |
|  $k$ |   Number of clusters |
|  $v$ |   View index |
|  $n_v$ | Number of views |
|  $D^{(v)}$ |   Dimension of data points in the view $v$ |
| $\mathbf{X}^{(v)}\in \mathbb{R}^{D^{(v)}\times N}$ |  Data matrix in the view $v$      |
| $\mathbf{C}^{(v)}\in \mathbb{R}^{N\times N}$ |  Representation matrix in the view $v$      |
| $\mathbf{C}^{*}\in \mathbb{R}^{N\times N}$ |  Centroid Representation matrix |
| $\mathbf{W}\in \mathbb{R}^{N\times N}$ |  Affinity matrix |


## Low-Rank Representation

Low-Rank Representation (LRR) try to find a low-rank representation matrix $\mathbf{C}\in\mathbb{R}^{N\times N}$ for input data $\mathbf{X}$, i.e.,

$$
\min\limits_{\mathbf{C}}\lVert \mathbf{C} \rVert_*\qquad \mathrm{s.t.}\quad \mathbf{X}=\mathbf{XC}
$$

<table><tr><td bgcolor="#f0f0f0">

Consider data vector $\mathbf{X}=\{\mathbf{x}_1,..., \mathbf{x}_n\}$, each of which can be represented by the linear combination of the dictionary $\mathbf{A}$,

$$
\mathbf{X}=\mathbf{AZ}
$$

where $\mathbf{Z}$ is the coefficient matrix with each $\mathbf{z}_i$ being the representation of $\mathbf{x}_i$.

The low-rank representation of data vectors $\mathbf{X}$ becomes the following problem,

$$
\begin{split}
\min\limits_{\mathbf{Z}}\quad &\lVert \mathbf{Z} \rVert_* \\
\mathrm{s.t.}\quad &\mathbf{X=AZ}
\end{split}
$$

Here, $\lVert\cdot\rVert_*$ denotes the nuclear norm of a matrix.


We assume that $\mathbf{X}=[\mathbf{X}_1,...,\mathbf{X}_k]$ each of which stands for data vector set with the same true label.

In order to segment the data into their respective subspaces, we need to compute an affinity matrix that encodes the pairwise affinities between data vectors. So we use the data $\mathbf{X}$ itself as the dictionary,

$$
\begin{split}
\min\limits_{\mathbf{Z}}\quad &\lVert \mathbf{Z} \rVert_* \\
\mathrm{s.t.}\quad &\mathbf{X=XZ}
\end{split}
$$

Note that there always exist feasible solutions even when the data sampling is insufficient, because a data vector can be used to represent itself in LRR.

**Theorem** Assume that the data sampling is sufficient such that $n_i >rank(\mathbf{X}_i)=d_i$. If the subspaces are independent then there exists an optimal solution $\mathbf{Z}^*$ that is block-diagonal:

$$
\mathbf{Z}^*=\begin{bmatrix}\mathbf{Z}_1^* & 0 & \cdots & 0\\ 0&\mathbf{Z}_2^*&\cdots&0\\\vdots&\vdots&\ddots&\vdots\\0&0&\cdots&\mathbf{Z}_k^*\end{bmatrix}
$$

</td></tr><table>