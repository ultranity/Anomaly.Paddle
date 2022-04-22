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

### F.interpolate
F.interpolate中可以使用scale_factor或者shape进行上\下采样，但scale_factor已知的情况下可以避免采样中的额外计算获得进5x提速

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

另外需要吐槽一下 paddle 官方给出的 [参考对应实现](https://github.com/PaddlePaddle/X2Paddle/blob/develop/docs/pytorch_project_convertor/API_docs/ops/torch.cholesky_solve.md) 选择把cholesky分解后的三角矩阵乘回去-_-，那不是分解了个寂寞。 希望paddle Hackathon 能够早点补上原生实现

### cdist
利用 broadcast 和 norm 实现 cdist,缺点是目前的 broadcast 机制会成倍消耗显存
使用条件判断适应2d/3d data，说起来没有atleast_nd这种函数还是不太方便
```python
def cdist(X, Y, p=2.0):
    dim = max(len(X.shape), len(Y.shape))
    if dim==3:
        if len(Y.shape)==2:
            Y = Y.unsqueeze(0)
        elif len(Y.shape)==1:
            Y = Y.unsqueeze(0).unsqueeze(0)
        else:
            assert len(Y.shape)==3
            assert Y.shape[0]=X.shape[0]
    #B, P, C = X.shape
    #1|B, R, C = Y.shape
    D = paddle.linalg.norm(X[:, :, None, :]-Y[None, :, :, :], p, axis=-1)
    return D
```
减少显存占用的写法：
```python
    def cdist(X, Y, p=2.0):
    #2d P, C = X.shape| R, C = Y.shape -> P,R
    P, C = X.shape
    R, C = Y.shape 
    #3d B, P, C = X.shape|1, R, C = Y.shape -> B, P,R
    #D = paddle.linalg.norm(X[:, None, :]-Y[None, :, :], axis=-1)
    D = paddle.zeros((P, R))
    for i in range(P):
        D[i,:] = paddle.linalg.norm(X[i, None, :]-Y, axis=-1)
        #D[i,:] = (X[i, None, :]-Y).square().sum(-1).sqrt_()
    return D
```



希望paddle Hackathon 能够早点补上原生实现(https://github.com/PaddlePaddle/community/blob/master/rfcs/APIs/20220316_api_design_for_cdist.md)

## 切片读写速度
Tensor 的切片读写调用slice相比pytorch要耗时很多，某些情况下还有额外的拷贝损耗，失去了一般思路上的预分配数组空间优势, 同时似乎也支持inplace操作
262.325 __getitem__  ..\dygraph\varbase_patch_methods.py:572
219.319 __setitem__  ..\dygraph\varbase_patch_methods.py:600
采用预分配+切片的方法
```python
    def cdist(X, Y, p=2.0):
    P, C = X.shape
    R, C = Y.shape 
    D = paddle.zeros((P, R))
    for i in range(P):
        D[i,:] = paddle.linalg.norm(X[i, None, :]-Y, axis=-1)
    return D
```
执行速度还不如concat，能慢一般左右
```python
def cdist(X, Y, p=2.0):
    P, C = X.shape
    R, C = Y.shape
    D = []
    for i in range(P):
        D.append(paddle.linalg.norm(X[i, None, :]-Y, axis=-1))
    return paddle.stack(D, 0)
```
%timeit
outs = paddle.randn((16,100,64,64))
outs -= outs.mean(0)
paddle.einsum('bchw, bdhw -> hwcd', outs, outs)
H,W,c,c = outs.shape
for i in range(H):
    outs[i,:,:,:] = paddle.inverse(outs[i,:,:,:] + eps * paddle.eye(c))
## Avgpool2D 默认表现不一致
paddle.nn.AvgPool2D 默认exclusive=True, 与pytorch对应的参数为exclusive=False