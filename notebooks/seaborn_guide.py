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
    # 🎨 Seaborn完全指南

    Seaborn是基于Matplotlib的Python数据可视化库，提供高级接口来绘制有吸引力且信息丰富的统计图形。

    ## 🎯 为什么使用Seaborn？

    - **美观**: 默认样式优雅，配色方案专业
    - **简洁**: 一行代码实现复杂可视化
    - **统计**: 内置统计估计和可视化
    - **集成**: 与Pandas DataFrame无缝集成
    - **主题**: 多种内置主题和调色板

    ## 📦 安装

    ```bash
    pip install seaborn
    # 或使用uv
    uv pip install seaborn
    ```

    当前版本要求: `seaborn>=0.13.2`
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 1️⃣ 基础设置

    ### 导入和配置

    Seaborn通常与以下库一起使用：
    - `matplotlib.pyplot` - 底层绘图
    - `pandas` - 数据处理
    - `numpy` - 数值计算
    """
    )
    return


@app.cell
def _():

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import pandas as pd
    import numpy as np
    import matplotlib.font_manager as fm

    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['font.serif'] = ['SimHei']
    import seaborn as sns
    sns.set_style("darkgrid",{"font.sans-serif":['simhei', 'Arial']})

    print("=" * 60)
    print("🎨 Seaborn基础设置")
    print("=" * 60)
    print(f"\nSeaborn版本: {sns.__version__}")
    print(f"Matplotlib版本: {plt.matplotlib.__version__}")
    print(f"Pandas版本: {pd.__version__}")
    print(f"NumPy版本: {np.__version__}")

    # 显示可用的样式
    print(f"\n可用样式: {sns.axes_style().keys()}")
    return np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 2️⃣ Seaborn样式和主题

    ### 样式类型

    | 样式 | 说明 | 适用场景 |
    |------|------|---------|
    | `darkgrid` | 深色网格 | 默认，适合大多数场景 |
    | `whitegrid` | 白色网格 | 清爽，适合演示 |
    | `dark` | 深色背景 | 无网格，简洁 |
    | `white` | 白色背景 | 最简洁 |
    | `ticks` | 带刻度 | 科学论文 |

    ### 调色板类型

    | 类型 | 函数 | 说明 |
    |------|------|------|
    | 分类 | `color_palette()` | 离散颜色 |
    | 连续 | `cubehelix_palette()` | 渐变色 |
    | 发散 | `diverging_palette()` | 双向渐变 |
    """
    )
    return


@app.cell
def _(np, pd, plt, sns):
    print("=" * 60)
    print("🎨 样式和调色板示例")
    print("=" * 60)

    # 创建示例数据
    _style_data = pd.DataFrame({
        'x': range(10),
        'y': np.random.randn(10).cumsum()
    })

    # 展示不同样式
    _styles = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']

    _fig1, _axes1 = plt.subplots(2, 3, figsize=(15, 8))
    _axes1 = _axes1.flatten()

    for _idx, _style in enumerate(_styles):
        sns.set_style(_style)
        _ax = _axes1[_idx]
        sns.lineplot(data=_style_data, x='x', y='y', ax=_ax)
        _ax.set_title(f'Style: {_style}')

    # 隐藏多余的子图
    _axes1[-1].axis('off')

    plt.tight_layout()
    plt.show()

    # 恢复默认样式
    sns.set_theme(style="darkgrid")

    # 展示调色板
    print("\n常用调色板:")
    _palettes = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind']

    _fig2, _axes2 = plt.subplots(2, 3, figsize=(15, 6))
    _axes2 = _axes2.flatten()

    for _idx2, _palette in enumerate(_palettes):
        _colors = sns.color_palette(_palette, 8)
        # palplot不支持ax参数，需要手动绘制
        _axes2[_idx2].imshow([_colors], aspect='auto')
        _axes2[_idx2].set_title(f'Palette: {_palette}')
        _axes2[_idx2].axis('off')

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 3️⃣ 关系图 (Relational Plots)

    ### 主要函数

    | 函数 | 说明 | 用途 |
    |------|------|------|
    | `scatterplot()` | 散点图 | 显示两个变量的关系 |
    | `lineplot()` | 折线图 | 显示趋势和时间序列 |
    | `relplot()` | 关系图(通用) | 支持分面和多种类型 |

    ### 关键参数

    - `x`, `y`: 数据列名
    - `hue`: 颜色分组
    - `size`: 大小分组
    - `style`: 样式分组
    - `data`: DataFrame数据源
    """
    )
    return


@app.cell
def _(np, pd, plt, sns):
    print("=" * 60)
    print("📊 关系图示例")
    print("=" * 60)

    # 创建示例数据
    np.random.seed(42)
    rel_data = pd.DataFrame({
        '身高': np.random.normal(170, 10, 100),
        '体重': np.random.normal(65, 15, 100),
        '年龄': np.random.randint(20, 60, 100),
        '性别': np.random.choice(['男', '女'], 100)
    })

    # 添加相关性
    rel_data['体重'] = rel_data['身高'] * 0.8 - 70 + np.random.normal(0, 5, 100)

    print(f"\n数据形状: {rel_data.shape}")
    print(f"\n前5行数据:\n{rel_data.head()}")

    # 1. 基础散点图
    _fig_rel1, _axes_rel1 = plt.subplots(2, 2, figsize=(14, 10))

    # 简单散点图
    sns.scatterplot(data=rel_data, x='身高', y='体重', ax=_axes_rel1[0, 0])
    _axes_rel1[0, 0].set_title('基础散点图')

    # 带颜色分组
    sns.scatterplot(data=rel_data, x='身高', y='体重', hue='性别', ax=_axes_rel1[0, 1])
    _axes_rel1[0, 1].set_title('按性别分组')

    # 带大小映射
    sns.scatterplot(data=rel_data, x='身高', y='体重', size='年龄',
                    hue='性别', ax=_axes_rel1[1, 0])
    _axes_rel1[1, 0].set_title('大小映射年龄')

    # 带样式
    sns.scatterplot(data=rel_data, x='身高', y='体重',
                    hue='性别', style='性别', s=100, ax=_axes_rel1[1, 1])
    _axes_rel1[1, 1].set_title('不同样式标记')

    plt.tight_layout()
    plt.show()

    # 2. 折线图
    time_data = pd.DataFrame({
        '日期': pd.date_range('2024-01-01', periods=30),
        '销售额': np.random.randn(30).cumsum() + 100,
        '类别': np.random.choice(['A', 'B'], 30)
    })

    _fig_rel2, _axes_rel2 = plt.subplots(1, 2, figsize=(14, 4))

    # 简单折线图
    sns.lineplot(data=time_data, x='日期', y='销售额', ax=_axes_rel2[0])
    _axes_rel2[0].set_title('销售趋势')
    _axes_rel2[0].tick_params(axis='x', rotation=45)

    # 分组折线图
    sns.lineplot(data=time_data, x='日期', y='销售额', hue='类别',
                 marker='o', ax=_axes_rel2[1])
    _axes_rel2[1].set_title('按类别分组')
    _axes_rel2[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 4️⃣ 分布图 (Distribution Plots)

    ### 主要函数

    | 函数 | 说明 | 用途 |
    |------|------|------|
    | `histplot()` | 直方图 | 显示数据分布 |
    | `kdeplot()` | 核密度估计图 | 平滑的分布曲线 |
    | `ecdfplot()` | 经验累积分布 | 累积概率 |
    | `rugplot()` | 地毯图 | 显示数据点位置 |
    | `distplot()` | 分布图(已弃用) | 使用histplot代替 |

    ### 关键参数

    - `kde`: 是否显示核密度估计
    - `bins`: 直方图箱数
    - `stat`: 统计类型(count, frequency, density, probability)
    """
    )
    return


@app.cell
def _(np, pd, plt, sns):
    print("=" * 60)
    print("📈 分布图示例")
    print("=" * 60)

    # 创建示例数据
    np.random.seed(42)
    dist_data = pd.DataFrame({
        '正态分布': np.random.normal(100, 15, 1000),
        '偏态分布': np.random.gamma(2, 2, 1000) * 10,
        '类别': np.random.choice(['A', 'B', 'C'], 1000)
    })

    print(f"\n数据统计:\n{dist_data.describe()}")

    # 1. 直方图
    _fig_dist1, _axes_dist1 = plt.subplots(2, 2, figsize=(14, 10))

    # 基础直方图
    sns.histplot(data=dist_data, x='正态分布', ax=_axes_dist1[0, 0])
    _axes_dist1[0, 0].set_title('基础直方图')

    # 带KDE的直方图
    sns.histplot(data=dist_data, x='正态分布', kde=True, ax=_axes_dist1[0, 1])
    _axes_dist1[0, 1].set_title('直方图 + KDE')

    # 分组直方图
    sns.histplot(data=dist_data, x='正态分布', hue='类别',
                 multiple='stack', ax=_axes_dist1[1, 0])
    _axes_dist1[1, 0].set_title('堆叠直方图')

    # 双变量直方图
    sns.histplot(data=dist_data, x='正态分布', y='偏态分布', ax=_axes_dist1[1, 1])
    _axes_dist1[1, 1].set_title('二维直方图')

    plt.tight_layout()
    plt.show()

    # 2. KDE图
    _fig_dist2, _axes_dist2 = plt.subplots(1, 2, figsize=(14, 4))

    # 单变量KDE
    sns.kdeplot(data=dist_data, x='正态分布', ax=_axes_dist2[0])
    _axes_dist2[0].set_title('核密度估计')

    # 分组KDE
    sns.kdeplot(data=dist_data, x='正态分布', hue='类别',
                fill=True, alpha=0.5, ax=_axes_dist2[1])
    _axes_dist2[1].set_title('分组KDE')

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 5️⃣ 分类图 (Categorical Plots)

    ### 主要函数

    | 函数 | 说明 | 用途 |
    |------|------|------|
    | `barplot()` | 条形图 | 显示均值和置信区间 |
    | `countplot()` | 计数图 | 显示类别频数 |
    | `boxplot()` | 箱线图 | 显示分布和异常值 |
    | `violinplot()` | 小提琴图 | 箱线图+KDE |
    | `stripplot()` | 散点分类图 | 显示所有数据点 |
    | `swarmplot()` | 蜂群图 | 不重叠的散点图 |
    | `pointplot()` | 点图 | 显示均值和置信区间 |

    ### 关键参数

    - `x`, `y`: 分类变量和数值变量
    - `hue`: 颜色分组
    - `order`: 类别顺序
    - `orient`: 方向(v垂直, h水平)
    """
    )
    return


@app.cell
def _(np, pd, plt, sns):
    print("=" * 60)
    print("📊 分类图示例")
    print("=" * 60)

    # 创建示例数据
    np.random.seed(42)
    cat_data = pd.DataFrame({
        '部门': np.random.choice(['销售', '技术', '市场', '人力'], 200),
        '薪资': np.random.normal(8000, 2000, 200),
        '工作年限': np.random.randint(1, 10, 200),
        '性别': np.random.choice(['男', '女'], 200)
    })

    # 调整薪资使其更真实
    cat_data.loc[cat_data['部门'] == '技术', '薪资'] += 2000
    cat_data.loc[cat_data['部门'] == '销售', '薪资'] += 1000
    cat_data['薪资'] = cat_data['薪资'].clip(lower=5000)

    print(f"\n数据形状: {cat_data.shape}")
    print(f"\n各部门人数:\n{cat_data['部门'].value_counts()}")

    # 1. 条形图和计数图
    _fig_cat1, _axes_cat1 = plt.subplots(2, 2, figsize=(14, 10))

    # 条形图 - 显示平均薪资
    sns.barplot(data=cat_data, x='部门', y='薪资', ax=_axes_cat1[0, 0])
    _axes_cat1[0, 0].set_title('各部门平均薪资')

    # 分组条形图
    sns.barplot(data=cat_data, x='部门', y='薪资', hue='性别', ax=_axes_cat1[0, 1])
    _axes_cat1[0, 1].set_title('按性别分组的平均薪资')

    # 计数图
    sns.countplot(data=cat_data, x='部门', ax=_axes_cat1[1, 0])
    _axes_cat1[1, 0].set_title('各部门人数')

    # 分组计数图
    sns.countplot(data=cat_data, x='部门', hue='性别', ax=_axes_cat1[1, 1])
    _axes_cat1[1, 1].set_title('按性别分组的人数')

    plt.tight_layout()
    plt.show()

    # 2. 箱线图和小提琴图
    _fig_cat2, _axes_cat2 = plt.subplots(2, 2, figsize=(14, 10))

    # 箱线图
    sns.boxplot(data=cat_data, x='部门', y='薪资', ax=_axes_cat2[0, 0])
    _axes_cat2[0, 0].set_title('薪资分布箱线图')

    # 分组箱线图
    sns.boxplot(data=cat_data, x='部门', y='薪资', hue='性别', ax=_axes_cat2[0, 1])
    _axes_cat2[0, 1].set_title('按性别分组的箱线图')

    # 小提琴图
    sns.violinplot(data=cat_data, x='部门', y='薪资', ax=_axes_cat2[1, 0])
    _axes_cat2[1, 0].set_title('薪资分布小提琴图')

    # 分组小提琴图
    sns.violinplot(data=cat_data, x='部门', y='薪资', hue='性别',
                   split=True, ax=_axes_cat2[1, 1])
    _axes_cat2[1, 1].set_title('分裂小提琴图')

    plt.tight_layout()
    plt.show()

    # 3. 散点分类图
    _fig_cat3, _axes_cat3 = plt.subplots(1, 2, figsize=(14, 5))

    # 散点分类图
    sns.stripplot(data=cat_data, x='部门', y='薪资', ax=_axes_cat3[0])
    _axes_cat3[0].set_title('散点分类图')

    # 蜂群图
    sns.swarmplot(data=cat_data.sample(100), x='部门', y='薪资', ax=_axes_cat3[1])
    _axes_cat3[1].set_title('蜂群图 (采样100个点)')

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 6️⃣ 矩阵图 (Matrix Plots)

    ### 主要函数

    | 函数 | 说明 | 用途 |
    |------|------|------|
    | `heatmap()` | 热力图 | 显示矩阵数据 |
    | `clustermap()` | 聚类热力图 | 带层次聚类的热力图 |

    ### 关键参数

    - `annot`: 是否显示数值
    - `fmt`: 数值格式
    - `cmap`: 颜色映射
    - `center`: 中心值
    - `vmin`, `vmax`: 值域范围
    - `linewidths`: 网格线宽度
    """
    )
    return


@app.cell
def _(np, pd, plt, sns):
    print("=" * 60)
    print("🔥 矩阵图示例")
    print("=" * 60)

    # 1. 相关系数热力图
    np.random.seed(42)
    corr_data = pd.DataFrame({
        '数学': np.random.randint(60, 100, 50),
        '物理': np.random.randint(60, 100, 50),
        '化学': np.random.randint(60, 100, 50),
        '英语': np.random.randint(60, 100, 50),
        '语文': np.random.randint(60, 100, 50)
    })

    # 添加相关性
    corr_data['物理'] = corr_data['数学'] * 0.7 + np.random.randint(-10, 10, 50)
    corr_data['化学'] = corr_data['数学'] * 0.5 + np.random.randint(-10, 10, 50)

    _correlation = corr_data.corr()

    print(f"\n相关系数矩阵:\n{_correlation}")

    _fig_mat1, _axes_mat1 = plt.subplots(1, 2, figsize=(14, 5))

    # 基础热力图
    sns.heatmap(_correlation, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=_axes_mat1[0])
    _axes_mat1[0].set_title('相关系数热力图')

    # 带掩码的热力图（只显示下三角）
    _mask = np.triu(np.ones_like(_correlation, dtype=bool))
    sns.heatmap(_correlation, mask=_mask, annot=True, fmt='.2f',
                cmap='RdYlGn', center=0, square=True,
                linewidths=1, ax=_axes_mat1[1])
    _axes_mat1[1].set_title('下三角热力图')

    plt.tight_layout()
    plt.show()

    # 2. 数据热力图
    _pivot_data = pd.DataFrame({
        '月份': ['1月', '2月', '3月', '4月', '5月', '6月'] * 4,
        '产品': ['A', 'A', 'A', 'A', 'A', 'A',
                'B', 'B', 'B', 'B', 'B', 'B',
                'C', 'C', 'C', 'C', 'C', 'C',
                'D', 'D', 'D', 'D', 'D', 'D'],
        '销量': np.random.randint(100, 500, 24)
    })

    _pivot_table = _pivot_data.pivot(index='产品', columns='月份', values='销量')

    plt.figure(figsize=(10, 4))
    sns.heatmap(_pivot_table, annot=True, fmt='d', cmap='YlOrRd',
                linewidths=0.5, cbar_kws={'label': '销量'})
    plt.title('产品月度销量热力图')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 7️⃣ 回归图 (Regression Plots)

    ### 主要函数

    | 函数 | 说明 | 用途 |
    |------|------|------|
    | `regplot()` | 回归图 | 显示线性回归 |
    | `lmplot()` | 线性模型图 | 支持分面的回归图 |
    | `residplot()` | 残差图 | 显示回归残差 |

    ### 关键参数

    - `x`, `y`: 变量
    - `order`: 多项式阶数
    - `logistic`: 是否逻辑回归
    - `lowess`: 是否使用局部加权回归
    - `scatter_kws`: 散点图参数
    - `line_kws`: 回归线参数
    """
    )
    return


@app.cell
def _(np, pd, plt, sns):
    print("=" * 60)
    print("📉 回归图示例")
    print("=" * 60)

    # 创建示例数据
    np.random.seed(42)
    reg_data = pd.DataFrame({
        '广告投入': np.random.uniform(10, 100, 100),
        '销售额': 0,
        '渠道': np.random.choice(['线上', '线下'], 100)
    })

    # 添加线性关系和噪声
    reg_data['销售额'] = (reg_data['广告投入'] * 2.5 + 50 +
                        np.random.normal(0, 20, 100))

    print(f"\n数据形状: {reg_data.shape}")
    print(f"\n相关系数: {reg_data['广告投入'].corr(reg_data['销售额']):.3f}")

    _fig_reg, _axes_reg = plt.subplots(2, 2, figsize=(14, 10))

    # 基础回归图
    sns.regplot(data=reg_data, x='广告投入', y='销售额', ax=_axes_reg[0, 0])
    _axes_reg[0, 0].set_title('线性回归图')

    # 二次回归
    sns.regplot(data=reg_data, x='广告投入', y='销售额',
                order=2, ax=_axes_reg[0, 1])
    _axes_reg[0, 1].set_title('二次回归')

    # 分组回归
    for _channel in reg_data['渠道'].unique():
        _subset = reg_data[reg_data['渠道'] == _channel]
        sns.regplot(data=_subset, x='广告投入', y='销售额',
                   label=_channel, ax=_axes_reg[1, 0])
    _axes_reg[1, 0].set_title('按渠道分组回归')
    _axes_reg[1, 0].legend()

    # 残差图
    sns.residplot(data=reg_data, x='广告投入', y='销售额', ax=_axes_reg[1, 1])
    _axes_reg[1, 1].set_title('残差图')
    _axes_reg[1, 1].axhline(0, color='red', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 8️⃣ 多变量图 (Multi-plot Grids)

    ### 主要函数

    | 函数 | 说明 | 用途 |
    |------|------|------|
    | `FacetGrid` | 分面网格 | 创建多个子图 |
    | `PairGrid` | 配对网格 | 变量两两配对 |
    | `pairplot()` | 配对图 | 快速创建配对图 |
    | `JointGrid` | 联合网格 | 双变量+边际分布 |
    | `jointplot()` | 联合图 | 快速创建联合图 |

    ### 关键参数

    - `col`, `row`: 分面变量
    - `hue`: 颜色分组
    - `kind`: 图表类型
    - `diag_kind`: 对角线图表类型
    """
    )
    return


@app.cell
def _(np, pd, plt, sns):
    print("=" * 60)
    print("🎭 多变量图示例")
    print("=" * 60)

    # 创建示例数据
    np.random.seed(42)
    iris_like = pd.DataFrame({
        '花瓣长度': np.concatenate([
            np.random.normal(1.5, 0.3, 50),
            np.random.normal(4.5, 0.5, 50),
            np.random.normal(6.0, 0.6, 50)
        ]),
        '花瓣宽度': np.concatenate([
            np.random.normal(0.3, 0.1, 50),
            np.random.normal(1.3, 0.2, 50),
            np.random.normal(2.0, 0.3, 50)
        ]),
        '花萼长度': np.concatenate([
            np.random.normal(5.0, 0.4, 50),
            np.random.normal(6.0, 0.5, 50),
            np.random.normal(6.5, 0.6, 50)
        ]),
        '花萼宽度': np.concatenate([
            np.random.normal(3.4, 0.4, 50),
            np.random.normal(2.8, 0.3, 50),
            np.random.normal(3.0, 0.3, 50)
        ]),
        '品种': ['A'] * 50 + ['B'] * 50 + ['C'] * 50
    })

    print(f"\n数据形状: {iris_like.shape}")
    print(f"\n各品种数量:\n{iris_like['品种'].value_counts()}")

    # 1. 配对图
    _pairplot_fig = sns.pairplot(iris_like, hue='品种',
                                 diag_kind='kde',
                                 plot_kws={'alpha': 0.6})
    _pairplot_fig.fig.suptitle('鸢尾花数据配对图', y=1.02)
    plt.show()

    # 2. 联合图
    _joint_fig = sns.jointplot(data=iris_like, x='花瓣长度', y='花瓣宽度',
                              hue='品种', kind='scatter', height=8)
    _joint_fig.fig.suptitle('花瓣长度vs宽度联合图', y=1.02)
    plt.show()

    # 3. 不同类型的联合图
    _kinds = ['scatter', 'kde', 'hex', 'reg']
    for _idx_multi, _kind in enumerate(_kinds):
        if _kind in ['hex', 'reg']:
            # hex和reg不支持hue参数
            _g = sns.jointplot(data=iris_like, x='花瓣长度', y='花瓣宽度',
                            kind=_kind, height=5)
        else:
            _g = sns.jointplot(data=iris_like, x='花瓣长度', y='花瓣宽度',
                            kind=_kind, hue='品种', height=5)
        _g.fig.suptitle(f'联合图: {_kind}', y=1.02)
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 9️⃣ 实战案例

    ### 案例1: 销售数据分析
    """
    )
    return


@app.cell
def _(np, pd, plt, sns):
    print("=" * 60)
    print("💼 案例1: 销售数据分析")
    print("=" * 60)

    # 创建销售数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=365)

    sales_data = pd.DataFrame({
        '日期': dates,
        '销售额': np.random.normal(10000, 2000, 365) +
                 np.sin(np.arange(365) * 2 * np.pi / 365) * 3000,
        '客户数': np.random.poisson(50, 365),
        '地区': np.random.choice(['北区', '南区', '东区', '西区'], 365),
        '产品类型': np.random.choice(['电子', '服装', '食品'], 365)
    })

    sales_data['月份'] = sales_data['日期'].dt.month
    sales_data['季度'] = sales_data['日期'].dt.quarter
    sales_data['客单价'] = sales_data['销售额'] / sales_data['客户数']

    print(f"\n数据形状: {sales_data.shape}")
    print(f"\n数据概览:\n{sales_data.head()}")
    print(f"\n统计信息:\n{sales_data.describe()}")

    # 综合分析图
    _fig_sales = plt.figure(figsize=(16, 12))
    _gs = _fig_sales.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. 销售额时间趋势
    _ax1 = _fig_sales.add_subplot(_gs[0, :])
    _monthly_sales = sales_data.groupby('月份')['销售额'].mean()
    sns.lineplot(x=_monthly_sales.index, y=_monthly_sales.values,
                marker='o', ax=_ax1)
    _ax1.set_title('月度平均销售额趋势', fontsize=14, fontweight='bold')
    _ax1.set_xlabel('月份')
    _ax1.set_ylabel('平均销售额')

    # 2. 地区销售分布
    _ax2 = _fig_sales.add_subplot(_gs[1, 0])
    sns.boxplot(data=sales_data, x='地区', y='销售额', ax=_ax2)
    _ax2.set_title('各地区销售额分布')
    _ax2.tick_params(axis='x', rotation=45)

    # 3. 产品类型销售
    _ax3 = _fig_sales.add_subplot(_gs[1, 1])
    sns.barplot(data=sales_data, x='产品类型', y='销售额',
               estimator=sum, ax=_ax3)
    _ax3.set_title('产品类型总销售额')
    _ax3.tick_params(axis='x', rotation=45)

    # 4. 季度对比
    _ax4 = _fig_sales.add_subplot(_gs[1, 2])
    sns.violinplot(data=sales_data, x='季度', y='销售额', ax=_ax4)
    _ax4.set_title('季度销售额分布')

    # 5. 客户数vs销售额
    _ax5 = _fig_sales.add_subplot(_gs[2, 0])
    sns.scatterplot(data=sales_data, x='客户数', y='销售额',
                   hue='地区', alpha=0.6, ax=_ax5)
    _ax5.set_title('客户数与销售额关系')

    # 6. 客单价分布
    _ax6 = _fig_sales.add_subplot(_gs[2, 1])
    sns.histplot(data=sales_data, x='客单价', kde=True, ax=_ax6)
    _ax6.set_title('客单价分布')

    # 7. 热力图
    _ax7 = _fig_sales.add_subplot(_gs[2, 2])
    _pivot_sales = sales_data.pivot_table(values='销售额',
                                   index='地区',
                                   columns='产品类型',
                                   aggfunc='mean')
    sns.heatmap(_pivot_sales, annot=True, fmt='.0f', cmap='YlOrRd', ax=_ax7)
    _ax7.set_title('地区-产品平均销售额')

    plt.suptitle('销售数据综合分析仪表板', fontsize=16, fontweight='bold', y=0.995)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🔟 常用技巧和最佳实践

    ### 1. 设置图表大小和样式

    ```python
    # 设置全局样式
    sns.set_theme(style="whitegrid", palette="pastel")

    # 设置单个图表大小
    plt.figure(figsize=(10, 6))

    # 设置字体大小
    sns.set_context("talk")  # paper, notebook, talk, poster
    ```

    ### 2. 保存图表

    ```python
    # 保存为PNG
    plt.savefig('plot.png', dpi=300, bbox_inches='tight')

    # 保存为PDF
    plt.savefig('plot.pdf', bbox_inches='tight')
    ```

    ### 3. 自定义颜色

    ```python
    # 使用调色板
    colors = sns.color_palette("husl", 8)

    # 自定义颜色
    custom_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    sns.set_palette(custom_colors)
    ```

    ### 4. 添加注释

    ```python
    # 添加文本
    ax.text(x, y, 'text', fontsize=12)

    # 添加箭头
    ax.annotate('point', xy=(x, y), xytext=(x2, y2),
               arrowprops=dict(arrowstyle='->'))
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📚 API速查表总结

    ### 图表类型选择指南

    | 数据类型 | 推荐图表 | Seaborn函数 |
    |---------|---------|------------|
    | 单变量分布 | 直方图、KDE | `histplot()`, `kdeplot()` |
    | 双变量关系 | 散点图、回归图 | `scatterplot()`, `regplot()` |
    | 分类对比 | 条形图、箱线图 | `barplot()`, `boxplot()` |
    | 时间序列 | 折线图 | `lineplot()` |
    | 相关性 | 热力图 | `heatmap()` |
    | 多变量 | 配对图、联合图 | `pairplot()`, `jointplot()` |

    ### 常用参数速查

    | 参数 | 说明 | 适用函数 |
    |------|------|---------|
    | `data` | DataFrame数据源 | 所有函数 |
    | `x`, `y` | 变量名 | 大多数函数 |
    | `hue` | 颜色分组 | 大多数函数 |
    | `size` | 大小映射 | 散点图 |
    | `style` | 样式分组 | 散点图、折线图 |
    | `palette` | 调色板 | 大多数函数 |
    | `ax` | Matplotlib轴对象 | 大多数函数 |

    ### 学习资源

    - [Seaborn官方文档](https://seaborn.pydata.org/)
    - [Seaborn教程](https://seaborn.pydata.org/tutorial.html)
    - [Seaborn示例库](https://seaborn.pydata.org/examples/index.html)
    - [Seaborn API参考](https://seaborn.pydata.org/api.html)
    """
    )
    return


if __name__ == "__main__":
    app.run()
