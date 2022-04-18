## Tricks & Logs

### 增量均值、协方差矩阵计算

待补

### 精度矩阵分解

由于马氏距离计算中需要用到精度矩阵（也即协方差矩阵的逆），实际训练过程可以直接保存精度矩阵而非协方差矩阵以减少后续冗余运算。

同时考虑到协方差矩阵是非负定实对称矩阵，求逆过程可以使用[cholesky分解](#cholesky_inverse)加速。

### einsum

einsum 实在是矩阵运算中的大杀器，可以减少很多转置、内积、外积的操作，但也有数据批量太大爆内存/显存的问题

但注意：paddle中 einsum 只能在动态图使用不能导出静态图，遇到这种需求还要手动实现


## Paddle 相关记录

### 设备间拷贝
> 见 [Issue](https://github.com/PaddlePaddle/Paddle/issues/41876)

Paddle 在 有 cuda 的设备上默认调用 GPU，实际运算中.cpu()等指定设备操作实际上没有效果，如设定某一 Tensor x 传输至 CPU 后，对其进行的任意运算仍将被自动传回 GPU 进行，对部分需要在 cpu 端进行的操作难以调度，不太符合在设备端运算的控制逻辑，device_guard 在动态图模式下无效， paddle.device.set_device ("cpu") 可用但仍旧添加了额外的逻辑负担。

### transpose 与 swapaxis
paddle中没有swapaxis，transpose的perm参数必须对应全部维度而不能部分交换，对可能出现不同维度处理的情况没法用 -1维之类的技巧而只能写分支判断

### in_place操作

### reshape 与 view

### 正交初始化
迁移自 pytorch的qr分解实现
```python
def orthogonal(rows,cols, gain=1):
    r"""return a (semi) orthogonal matrix, as
    described in `Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks` - Saxe, A. et al. (2013). 
    Args:
        rows: rows
        cols: cols
        gain: optional scaling factor
    Examples:
        >>> orthogonal_(5, 3)
    """
    flattened = paddle.randn((rows,cols))

    if rows < cols:
        flattened = flattened.T

    # Compute the qr factorization
    q, r = paddle.linalg.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = paddle.diag(r, 0)
    q *= d.sign()

    if rows < cols:
        q = q.T
    
    q *= gain
    return q
```

另一种方式是利用svd分解：
```python
def svd_orthogonal(fin,fout, use_paddle=False):
    assert fin > fout, 'fin > fout'
    if use_paddle:
        X = paddle.rand((fout, fin))
        U, _, Vt = paddle.linalg.svd(X, full_matrices=False)
        #print(Vt.shape)
        #print(paddle.allclose(Vt@Vt.T, paddle.eye(Vt.shape[0])))
    else:
        X = np.random.random((fout, fin))
        U, _, Vt = np.linalg.svd(X, full_matrices=False)
        #print(Vt.shape)
        #print(np.allclose((Vt@ Vt.T), np.eye(Vt.shape[0])))
    W = paddle.to_tensor(Vt, dtype=paddle.float32).T
    return W
```
### cholesky_inverse

矩阵的对称性可以用来加速大规模矩阵运算，特别是矩阵求逆这种复杂运算

然鹅paddle 中实现了cholesky分解，但没有相应的cholesky_inverse 和 cholesky_solve 算子 原生实现，可以用triangular_solve先凑合，也能起到加速效果
```python
def cholesky_inverse(input, upper=False, out=None) :
    u = paddle.cholesky(input, upper)
    ui = paddle.linalg.triangular_solve(u, paddle.eye(u.shape[-1]), upper=upper)
    if len(u.shape)==2:
        uit = ui.T
    elif len(u.shape)==3:
        uit =paddle.transpose(ui, perm=(0, 2, 1))
    elif len(u.shape)==4:
        uit = paddle.transpose(ui, perm=(0, 1, 3, 2))
    out = ui@uit if upper else uit@ui
    return out
```


更进一步，利用cholesky分解的对称三角分解性质，理论上还可以减少内存、显存占用，然而numpy、pytorch等python下的框架都很少做这种优化，在BLAS层面才有

另外需要吐槽一下 paddle 官方给出的 [参考对应实现](https://github.com/PaddlePaddle/X2Paddle/blob/develop/docs/pytorch_project_convertor/API_docs/ops/torch.cholesky_solve.md) 选择把cholesky分解后的三角矩阵乘回去-_-，那不是分解了个寂寞