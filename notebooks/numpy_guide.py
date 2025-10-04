import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # 📊 NumPy完全指南

    NumPy (Numerical Python) 是Python科学计算的基础库，提供高性能的多维数组对象和处理这些数组的工具。

    ## 🎯 为什么使用NumPy？

    - **性能**: 比Python原生列表快10-100倍
    - **内存效率**: 连续内存存储，占用更少空间
    - **向量化操作**: 避免显式循环，代码更简洁
    - **广泛支持**: 几乎所有科学计算库都基于NumPy

    ## 📦 安装

    ```bash
    pip install numpy
    # 或使用uv
    uv pip install numpy
    ```

    当前版本要求: `numpy>=2.3.3`
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 1️⃣ 基础概念

    ### 核心对象: ndarray

    NumPy的核心是`ndarray`（N-dimensional array）对象，它是一个多维数组。

    ### 重要属性

    | 属性 | 说明 | 示例 |
    |------|------|------|
    | `ndarray.shape` | 数组维度 | `(3, 4)` 表示3行4列 |
    | `ndarray.dtype` | 元素类型 | `int64`, `float32`, `bool` |
    | `ndarray.ndim` | 维度数量 | `2` 表示二维数组 |
    | `ndarray.size` | 元素总数 | `12` (3×4) |
    | `ndarray.itemsize` | 每个元素字节数 | `8` (int64) |
    | `ndarray.nbytes` | 总字节数 | `size × itemsize` |
    """
    )
    return


@app.cell
def _():
    import numpy as np

    print("=" * 60)
    print("📊 NumPy数组基础")
    print("=" * 60)

    # 创建数组
    arr = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])

    print(f"\n数组内容:\n{arr}")
    print(f"\n形状 (shape): {arr.shape}")
    print(f"维度 (ndim): {arr.ndim}")
    print(f"大小 (size): {arr.size}")
    print(f"数据类型 (dtype): {arr.dtype}")
    print(f"每个元素字节数 (itemsize): {arr.itemsize}")
    print(f"总字节数 (nbytes): {arr.nbytes}")

    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 2️⃣ 数组创建方法

    ### 常用创建函数

    | 函数 | 说明 | 示例 |
    |------|------|------|
    | `np.array()` | 从列表/元组创建 | `np.array([1, 2, 3])` |
    | `np.zeros()` | 全0数组 | `np.zeros((3, 4))` |
    | `np.ones()` | 全1数组 | `np.ones((2, 3))` |
    | `np.full()` | 指定值填充 | `np.full((2, 2), 7)` |
    | `np.eye()` | 单位矩阵 | `np.eye(3)` |
    | `np.arange()` | 等差数列 | `np.arange(0, 10, 2)` |
    | `np.linspace()` | 线性等分 | `np.linspace(0, 1, 5)` |
    | `np.random.rand()` | 均匀分布随机 | `np.random.rand(3, 3)` |
    | `np.random.randn()` | 标准正态分布 | `np.random.randn(3, 3)` |
    | `np.random.randint()` | 随机整数 | `np.random.randint(0, 10, (3, 3))` |
    """
    )
    return


@app.cell
def _(np):
    print("=" * 60)
    print("🔨 数组创建示例")
    print("=" * 60)

    # 从列表创建
    arr1 = np.array([1, 2, 3, 4, 5])
    print(f"\n从列表创建:\n{arr1}")

    # 全0数组
    zeros = np.zeros((2, 3))
    print(f"\n全0数组 (2×3):\n{zeros}")

    # 全1数组
    ones = np.ones((3, 2))
    print(f"\n全1数组 (3×2):\n{ones}")

    # 指定值填充
    full = np.full((2, 4), 7)
    print(f"\n填充7 (2×4):\n{full}")

    # 单位矩阵
    eye = np.eye(3)
    print(f"\n单位矩阵 (3×3):\n{eye}")

    # 等差数列
    arange = np.arange(0, 10, 2)
    print(f"\n等差数列 (0到10，步长2):\n{arange}")

    # 线性等分
    linspace = np.linspace(0, 1, 5)
    print(f"\n线性等分 (0到1，5个点):\n{linspace}")

    # 随机数组
    np.random.seed(42)  # 设置随机种子以便复现
    rand = np.random.rand(2, 3)
    print(f"\n均匀分布随机数 (2×3):\n{rand}")

    randn = np.random.randn(2, 3)
    print(f"\n标准正态分布 (2×3):\n{randn}")

    randint = np.random.randint(0, 10, (2, 3))
    print(f"\n随机整数 [0, 10) (2×3):\n{randint}")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 3️⃣ 数组索引与切片

    ### 索引方式

    | 方式 | 说明 | 示例 |
    |------|------|------|
    | 基本索引 | 单个元素 | `arr[0, 1]` |
    | 切片 | 范围选择 | `arr[1:3, :]` |
    | 布尔索引 | 条件筛选 | `arr[arr > 5]` |
    | 花式索引 | 整数数组索引 | `arr[[0, 2], [1, 3]]` |
    """
    )
    return


@app.cell
def _(np):
    print("=" * 60)
    print("🎯 数组索引与切片")
    print("=" * 60)

    arr_idx = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12]])

    print(f"\n原始数组:\n{arr_idx}")

    # 基本索引
    print(f"\narr[0, 1] = {arr_idx[0, 1]}")  # 第0行第1列
    print(f"arr[2, 3] = {arr_idx[2, 3]}")  # 第2行第3列

    # 切片
    print(f"\narr[1:3, :] (第1-2行，所有列):\n{arr_idx[1:3, :]}")
    print(f"\narr[:, 1:3] (所有行，第1-2列):\n{arr_idx[:, 1:3]}")
    print(f"\narr[::2, ::2] (隔行隔列):\n{arr_idx[::2, ::2]}")

    # 布尔索引
    mask = arr_idx > 6
    print(f"\n布尔掩码 (arr > 6):\n{mask}")
    print(f"\n筛选结果 (arr[arr > 6]):\n{arr_idx[mask]}")

    # 花式索引
    fancy = arr_idx[[0, 2], [1, 3]]
    print(f"\n花式索引 arr[[0, 2], [1, 3]]:\n{fancy}")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 4️⃣ 数组运算

    ### 算术运算

    | 运算 | 符号 | 函数 | 说明 |
    |------|------|------|------|
    | 加法 | `+` | `np.add()` | 逐元素相加 |
    | 减法 | `-` | `np.subtract()` | 逐元素相减 |
    | 乘法 | `*` | `np.multiply()` | 逐元素相乘 |
    | 除法 | `/` | `np.divide()` | 逐元素相除 |
    | 幂运算 | `**` | `np.power()` | 逐元素求幂 |
    | 矩阵乘法 | `@` | `np.dot()`, `np.matmul()` | 矩阵乘法 |
    """
    )
    return


@app.cell
def _(np):
    print("=" * 60)
    print("➕ 数组运算")
    print("=" * 60)

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])

    print(f"\n数组 a:\n{a}")
    print(f"\n数组 b:\n{b}")

    # 算术运算
    print(f"\na + b:\n{a + b}")
    print(f"\na - b:\n{a - b}")
    print(f"\na * b (逐元素):\n{a * b}")
    print(f"\na / b:\n{a / b}")
    print(f"\na ** 2:\n{a ** 2}")

    # 矩阵乘法
    print(f"\na @ b (矩阵乘法):\n{a @ b}")
    print(f"\nnp.dot(a, b):\n{np.dot(a, b)}")

    # 标量运算
    print(f"\na + 10:\n{a + 10}")
    print(f"\na * 2:\n{a * 2}")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 5️⃣ 统计函数

    ### 常用统计方法

    | 函数 | 说明 | 示例 |
    |------|------|------|
    | `np.sum()` | 求和 | `arr.sum()` |
    | `np.mean()` | 平均值 | `arr.mean()` |
    | `np.std()` | 标准差 | `arr.std()` |
    | `np.var()` | 方差 | `arr.var()` |
    | `np.min()` | 最小值 | `arr.min()` |
    | `np.max()` | 最大值 | `arr.max()` |
    | `np.argmin()` | 最小值索引 | `arr.argmin()` |
    | `np.argmax()` | 最大值索引 | `arr.argmax()` |
    | `np.median()` | 中位数 | `np.median(arr)` |
    | `np.percentile()` | 百分位数 | `np.percentile(arr, 50)` |
    """
    )
    return


@app.cell
def _(np):
    print("=" * 60)
    print("📈 统计函数")
    print("=" * 60)

    data = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12]])

    print(f"\n数据:\n{data}")

    # 全局统计
    print(f"\n总和: {data.sum()}")
    print(f"平均值: {data.mean():.2f}")
    print(f"标准差: {data.std():.2f}")
    print(f"方差: {data.var():.2f}")
    print(f"最小值: {data.min()}")
    print(f"最大值: {data.max()}")

    # 按轴统计
    print(f"\n按行求和 (axis=1): {data.sum(axis=1)}")
    print(f"按列求和 (axis=0): {data.sum(axis=0)}")
    print(f"按列平均 (axis=0): {data.mean(axis=0)}")

    # 索引
    print(f"\n最小值索引: {data.argmin()}")
    print(f"最大值索引: {data.argmax()}")
    print(f"按列最大值索引: {data.argmax(axis=0)}")

    # 其他统计
    print(f"\n中位数: {np.median(data)}")
    print(f"25%分位数: {np.percentile(data, 25)}")
    print(f"75%分位数: {np.percentile(data, 75)}")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 6️⃣ 形状操作

    ### 形状变换函数

    | 函数 | 说明 | 示例 |
    |------|------|------|
    | `reshape()` | 改变形状 | `arr.reshape(3, 4)` |
    | `flatten()` | 展平为1D | `arr.flatten()` |
    | `ravel()` | 展平(视图) | `arr.ravel()` |
    | `transpose()` | 转置 | `arr.T` 或 `arr.transpose()` |
    | `squeeze()` | 删除长度为1的维度 | `arr.squeeze()` |
    | `expand_dims()` | 增加维度 | `np.expand_dims(arr, axis=0)` |
    """
    )
    return


@app.cell
def _(np):
    print("=" * 60)
    print("🔄 形状操作")
    print("=" * 60)

    shape_arr = np.arange(12)
    print(f"\n原始数组: {shape_arr}")
    print(f"形状: {shape_arr.shape}")

    # reshape
    reshaped = shape_arr.reshape(3, 4)
    print(f"\nreshape(3, 4):\n{reshaped}")

    # 自动推断维度
    auto_reshape = shape_arr.reshape(2, -1)
    print(f"\nreshape(2, -1) [自动推断]:\n{auto_reshape}")

    # 转置
    print(f"\n转置 .T:\n{reshaped.T}")

    # 展平
    print(f"\nflatten(): {reshaped.flatten()}")
    print(f"ravel(): {reshaped.ravel()}")

    # 增加/删除维度
    expanded = np.expand_dims(shape_arr, axis=0)
    print(f"\nexpand_dims(axis=0) 形状: {expanded.shape}")

    squeezed = expanded.squeeze()
    print(f"squeeze() 形状: {squeezed.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 7️⃣ 数组拼接与分割

    ### 拼接函数

    | 函数 | 说明 | 示例 |
    |------|------|------|
    | `np.concatenate()` | 沿指定轴拼接 | `np.concatenate([a, b], axis=0)` |
    | `np.vstack()` | 垂直堆叠(行) | `np.vstack([a, b])` |
    | `np.hstack()` | 水平堆叠(列) | `np.hstack([a, b])` |
    | `np.stack()` | 沿新轴堆叠 | `np.stack([a, b], axis=0)` |

    ### 分割函数

    | 函数 | 说明 | 示例 |
    |------|------|------|
    | `np.split()` | 沿指定轴分割 | `np.split(arr, 3, axis=0)` |
    | `np.vsplit()` | 垂直分割 | `np.vsplit(arr, 2)` |
    | `np.hsplit()` | 水平分割 | `np.hsplit(arr, 3)` |
    """
    )
    return


@app.cell
def _(np):
    print("=" * 60)
    print("🔗 数组拼接与分割")
    print("=" * 60)

    arr_a = np.array([[1, 2], [3, 4]])
    arr_b = np.array([[5, 6], [7, 8]])

    print(f"\n数组 a:\n{arr_a}")
    print(f"\n数组 b:\n{arr_b}")

    # 拼接
    vstack_result = np.vstack([arr_a, arr_b])
    print(f"\nvstack (垂直堆叠):\n{vstack_result}")

    hstack_result = np.hstack([arr_a, arr_b])
    print(f"\nhstack (水平堆叠):\n{hstack_result}")

    concat_0 = np.concatenate([arr_a, arr_b], axis=0)
    print(f"\nconcatenate(axis=0):\n{concat_0}")

    concat_1 = np.concatenate([arr_a, arr_b], axis=1)
    print(f"\nconcatenate(axis=1):\n{concat_1}")

    # 分割
    split_arr = np.arange(12).reshape(4, 3)
    print(f"\n待分割数组:\n{split_arr}")

    vsplit_result = np.vsplit(split_arr, 2)
    print(f"\nvsplit(2) 结果:")
    for i, part in enumerate(vsplit_result):
        print(f"  部分{i+1}:\n{part}")

    hsplit_result = np.hsplit(split_arr, 3)
    print(f"\nhsplit(3) 结果:")
    for i, part in enumerate(hsplit_result):
        print(f"  部分{i+1}:\n{part}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 8️⃣ 数学函数

    ### 通用函数(ufunc)

    | 类别 | 函数 | 说明 |
    |------|------|------|
    | **三角函数** | `sin`, `cos`, `tan` | 三角函数 |
    | | `arcsin`, `arccos`, `arctan` | 反三角函数 |
    | **指数对数** | `exp`, `log`, `log10`, `log2` | 指数和对数 |
    | | `sqrt`, `square` | 平方根和平方 |
    | **取整** | `round`, `floor`, `ceil` | 四舍五入、向下、向上取整 |
    | **符号** | `abs`, `sign` | 绝对值和符号 |
    | **比较** | `maximum`, `minimum` | 逐元素最大/最小值 |
    """
    )
    return


@app.cell
def _(np):
    print("=" * 60)
    print("🧮 数学函数")
    print("=" * 60)

    angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
    print(f"\n角度 (弧度): {angles}")
    print(f"sin: {np.sin(angles)}")
    print(f"cos: {np.cos(angles)}")

    # 指数对数
    nums = np.array([1, 2, 4, 8, 16])
    print(f"\n数字: {nums}")
    print(f"exp: {np.exp(nums)[:3]}...")  # 只显示前3个
    print(f"log: {np.log(nums)}")
    print(f"log2: {np.log2(nums)}")
    print(f"sqrt: {np.sqrt(nums)}")

    # 取整
    decimals = np.array([1.2, 2.5, 3.7, -1.8, -2.3])
    print(f"\n小数: {decimals}")
    print(f"round: {np.round(decimals)}")
    print(f"floor: {np.floor(decimals)}")
    print(f"ceil: {np.ceil(decimals)}")

    # 符号和绝对值
    mixed = np.array([-3, -1, 0, 2, 5])
    print(f"\n混合数: {mixed}")
    print(f"abs: {np.abs(mixed)}")
    print(f"sign: {np.sign(mixed)}")

    # 比较
    x = np.array([1, 5, 3, 8, 2])
    y = np.array([4, 2, 6, 7, 9])
    print(f"\nx: {x}")
    print(f"y: {y}")
    print(f"maximum(x, y): {np.maximum(x, y)}")
    print(f"minimum(x, y): {np.minimum(x, y)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 9️⃣ 线性代数

    ### 线性代数函数

    | 函数 | 说明 | 示例 |
    |------|------|------|
    | `np.dot()` | 点积/矩阵乘法 | `np.dot(a, b)` |
    | `np.linalg.inv()` | 矩阵求逆 | `np.linalg.inv(a)` |
    | `np.linalg.det()` | 行列式 | `np.linalg.det(a)` |
    | `np.linalg.eig()` | 特征值和特征向量 | `np.linalg.eig(a)` |
    | `np.linalg.norm()` | 范数 | `np.linalg.norm(a)` |
    | `np.linalg.solve()` | 求解线性方程组 | `np.linalg.solve(A, b)` |
    | `np.trace()` | 矩阵的迹 | `np.trace(a)` |
    """
    )
    return


@app.cell
def _(np):
    print("=" * 60)
    print("🔢 线性代数")
    print("=" * 60)

    # 矩阵运算
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    print(f"\n矩阵 A:\n{A}")
    print(f"\n矩阵 B:\n{B}")

    # 矩阵乘法
    print(f"\nA @ B:\n{A @ B}")

    # 行列式
    print(f"\ndet(A): {np.linalg.det(A):.2f}")

    # 矩阵的迹
    print(f"trace(A): {np.trace(A)}")

    # 矩阵求逆
    A_inv = np.linalg.inv(A)
    print(f"\ninv(A):\n{A_inv}")
    print(f"\nA @ inv(A) (应该是单位矩阵):\n{A @ A_inv}")

    # 特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"\n特征值: {eigenvalues}")
    print(f"特征向量:\n{eigenvectors}")

    # 范数
    v = np.array([3, 4])
    print(f"\n向量 v: {v}")
    print(f"L2范数 (欧几里得距离): {np.linalg.norm(v):.2f}")
    print(f"L1范数: {np.linalg.norm(v, ord=1):.2f}")

    # 求解线性方程组 Ax = b
    A_eq = np.array([[3, 1], [1, 2]])
    b_eq = np.array([9, 8])
    x_solution = np.linalg.solve(A_eq, b_eq)
    print(f"\n求解 Ax = b:")
    print(f"A:\n{A_eq}")
    print(f"b: {b_eq}")
    print(f"解 x: {x_solution}")
    print(f"验证 Ax: {A_eq @ x_solution}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🔟 广播机制

    NumPy的广播(Broadcasting)允许不同形状的数组进行运算。

    ### 广播规则

    1. 如果两个数组维度不同，较小维度的数组会在前面补1
    2. 如果两个数组在某个维度上的大小不同，且其中一个为1，则该维度会被"拉伸"
    3. 如果两个数组在某个维度上大小不同且都不为1，则报错

    ### 示例

    ```python
    # (3, 4) + (4,) → (3, 4) + (1, 4) → (3, 4)
    # (3, 1) * (1, 4) → (3, 4)
    ```
    """
    )
    return


@app.cell
def _(np):
    print("=" * 60)
    print("📡 广播机制")
    print("=" * 60)

    # 示例1: 数组 + 标量
    arr_broadcast = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"\n数组:\n{arr_broadcast}")
    print(f"\n数组 + 10:\n{arr_broadcast + 10}")

    # 示例2: 2D数组 + 1D数组
    row_vec = np.array([10, 20, 30])
    print(f"\n行向量: {row_vec}")
    print(f"\n数组 + 行向量:\n{arr_broadcast + row_vec}")

    # 示例3: 列向量 + 行向量
    col_vec = np.array([[1], [2], [3]])
    row_vec2 = np.array([10, 20, 30])
    print(f"\n列向量:\n{col_vec}")
    print(f"行向量: {row_vec2}")
    print(f"\n列向量 + 行向量:\n{col_vec + row_vec2}")

    # 示例4: 实际应用 - 标准化
    data_norm = np.random.randn(3, 4)
    print(f"\n原始数据:\n{data_norm}")

    mean = data_norm.mean(axis=0)
    std = data_norm.std(axis=0)
    normalized = (data_norm - mean) / std

    print(f"\n每列均值: {mean}")
    print(f"每列标准差: {std}")
    print(f"\n标准化后:\n{normalized}")
    print(f"标准化后均值: {normalized.mean(axis=0)}")
    print(f"标准化后标准差: {normalized.std(axis=0)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 1️⃣1️⃣ 实战案例

    ### 案例1: 图像处理基础
    """
    )
    return


@app.cell
def _(np):
    print("=" * 60)
    print("🖼️  案例1: 图像处理")
    print("=" * 60)

    # 创建一个模拟的灰度图像 (8x8)
    image = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
    print(f"\n原始图像 (8x8):\n{image}")

    # 图像归一化到 [0, 1]
    image_normalized = image / 255.0
    print(f"\n归一化图像:\n{image_normalized}")

    # 图像翻转
    flipped_v = np.flipud(image)  # 垂直翻转
    flipped_h = np.fliplr(image)  # 水平翻转
    print(f"\n垂直翻转:\n{flipped_v}")

    # 图像旋转90度
    rotated = np.rot90(image)
    print(f"\n旋转90度:\n{rotated}")

    # 提取图像块
    patch = image[2:5, 2:5]
    print(f"\n提取图像块 [2:5, 2:5]:\n{patch}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### 案例2: 数据分析""")
    return


@app.cell
def _(np):
    print("=" * 60)
    print("📊 案例2: 数据分析")
    print("=" * 60)

    # 模拟学生成绩数据 (100个学生，5门课程)
    np.random.seed(42)
    scores = np.random.randint(60, 100, (100, 5))

    print(f"数据形状: {scores.shape}")
    print(f"前5个学生成绩:\n{scores[:5]}")

    # 统计分析
    print(f"\n每门课程平均分: {scores.mean(axis=0)}")
    print(f"每门课程最高分: {scores.max(axis=0)}")
    print(f"每门课程最低分: {scores.min(axis=0)}")
    print(f"每门课程标准差: {scores.std(axis=0)}")

    # 学生总分和平均分
    total_scores = scores.sum(axis=1)
    avg_scores = scores.mean(axis=1)

    print(f"\n前10名学生总分: {total_scores[:10]}")
    print(f"前10名学生平均分: {avg_scores[:10]}")

    # 找出优秀学生 (平均分 >= 85)
    excellent = avg_scores >= 85
    print(f"\n优秀学生数量: {excellent.sum()}")
    print(f"优秀学生比例: {excellent.sum() / len(scores) * 100:.1f}%")

    # 找出每门课程的最高分学生
    top_students = scores.argmax(axis=0)
    print(f"\n每门课程最高分学生索引: {top_students}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📚 总结

    ### NumPy核心优势

    1. **性能**: C语言实现，比Python列表快10-100倍
    2. **内存效率**: 连续内存存储，节省空间
    3. **向量化**: 避免显式循环，代码简洁高效
    4. **广播机制**: 灵活处理不同形状的数组
    5. **丰富的函数库**: 涵盖数学、统计、线性代数等

    ### 最佳实践

    - ✅ 使用向量化操作代替循环
    - ✅ 利用广播机制简化代码
    - ✅ 注意数组的视图(view)和副本(copy)
    - ✅ 选择合适的数据类型节省内存
    - ✅ 使用`axis`参数进行维度操作

    ### 学习资源

    - [NumPy官方文档](https://numpy.org/doc/)
    - [NumPy用户指南](https://numpy.org/doc/stable/user/index.html)
    - [NumPy API参考](https://numpy.org/doc/stable/reference/index.html)
    """
    )
    return


if __name__ == "__main__":
    app.run()
