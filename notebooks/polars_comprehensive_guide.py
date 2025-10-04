import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # ⚡ Polars完全指南

    ## 📚 什么是Polars？

    **Polars**是一个用Rust编写的极速DataFrame库，专为高性能数据处理而设计。它是Pandas的现代化替代品，提供了更快的速度和更低的内存占用。

    ### 核心特点

    - **⚡ 极速性能**：基于Rust和Apache Arrow，比Pandas快10-100倍
    - **🧠 内存高效**：优化的内存使用，处理大数据集更轻松
    - **🔄 惰性求值**：支持查询优化，只在需要时计算
    - **🎯 表达式API**：强大而优雅的表达式语法
    - **🔗 并行处理**：自动利用多核CPU
    - **📊 类型安全**：严格的类型系统，减少运行时错误
    - **🌐 多语言支持**：Python、Rust、Node.js等

    ### 为什么选择Polars？

    | 特性 | Polars | Pandas |
    |------|--------|--------|
    | **性能** | 🚀 极快 | 🐢 较慢 |
    | **内存** | 💚 高效 | 💛 一般 |
    | **并行** | ✅ 自动 | ❌ 需手动 |
    | **惰性求值** | ✅ 支持 | ❌ 不支持 |
    | **类型系统** | ✅ 严格 | ⚠️ 宽松 |
    | **生态系统** | 🌱 新兴 | 🌳 成熟 |

    ### 本指南内容

    1. **基础概念**：DataFrame和LazyFrame
    2. **数据创建**：从各种数据源创建
    3. **数据查看**：查看和检查数据
    4. **数据选择**：表达式API
    5. **数据清洗**：处理缺失值和重复值
    6. **数据转换**：排序、映射、应用函数
    7. **数据聚合**：分组和聚合操作
    8. **数据合并**：连接和拼接
    9. **惰性求值**：LazyFrame和查询优化
    10. **性能对比**：Polars vs Pandas
    11. **迁移指南**：从Pandas迁移到Polars
    12. **实战案例**：完整的数据分析流程
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import polars as pl
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import time

    print(f"✅ Polars版本: {pl.__version__}")
    print(f"✅ Pandas版本: {pd.__version__}")
    print(f"✅ NumPy版本: {np.__version__}")
    return datetime, np, pd, pl, time, timedelta


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📊 1. 核心数据结构

    Polars有两个主要的数据结构：

    ### DataFrame（即时求值）

    **DataFrame**是Polars的主要数据结构，类似于Pandas的DataFrame。

    **特点**：
    - 即时执行操作
    - 适合交互式分析
    - 类似Pandas的使用方式

    ### LazyFrame（惰性求值）

    **LazyFrame**是Polars的惰性执行版本，不会立即执行操作。

    **特点**：
    - 延迟执行，构建查询计划
    - 自动优化查询
    - 适合大数据集和复杂查询
    - 需要调用`.collect()`才执行
    """
    )
    return


@app.cell
def _(pl):
    print("=" * 60)
    print("📊 DataFrame vs LazyFrame")
    print("=" * 60)

    # 1. 创建DataFrame
    df_eager_pl = pl.DataFrame({
        '姓名': ['张三', '李四', '王五', '赵六', '钱七'],
        '年龄': [25, 30, 35, 28, 32],
        '城市': ['北京', '上海', '广州', '深圳', '杭州'],
        '薪资': [15000, 18000, 16000, 17000, 19000]
    }, strict=False)

    print("\n1️⃣ DataFrame（即时求值）:")
    print(df_eager_pl)
    print(f"   类型: {type(df_eager_pl)}")

    # 2. 创建LazyFrame
    df_lazy_pl = pl.DataFrame({
        '姓名': ['张三', '李四', '王五', '赵六', '钱七'],
        '年龄': [25, 30, 35, 28, 32],
        '城市': ['北京', '上海', '广州', '深圳', '杭州'],
        '薪资': [15000, 18000, 16000, 17000, 19000]
    }, strict=False).lazy()

    print("\n2️⃣ LazyFrame（惰性求值）:")
    print(df_lazy_pl)
    print(f"   类型: {type(df_lazy_pl)}")

    # 3. LazyFrame需要collect()才执行
    print("\n3️⃣ LazyFrame执行查询:")
    result_pl = df_lazy_pl.filter(pl.col('年龄') > 28).collect()
    print(result_pl)

    # 4. DataFrame和LazyFrame互转
    print("\n4️⃣ DataFrame和LazyFrame互转:")
    lazy_from_eager_pl = df_eager_pl.lazy()
    print(f"   DataFrame -> LazyFrame: {type(lazy_from_eager_pl)}")

    eager_from_lazy_pl = df_lazy_pl.collect()
    print(f"   LazyFrame -> DataFrame: {type(eager_from_lazy_pl)}")

    return df_eager_pl, df_lazy_pl, eager_from_lazy_pl, lazy_from_eager_pl, result_pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🎨 2. 数据创建

    Polars支持从多种数据源创建DataFrame。
    """
    )
    return


@app.cell
def _(np, pd, pl):
    print("=" * 60)
    print("🎨 数据创建示例")
    print("=" * 60)

    # 1. 从字典创建
    print("\n1️⃣ 从字典创建DataFrame:")
    df_from_dict_pl = pl.DataFrame({
        '产品': ['A', 'B', 'C', 'D'],
        '销售额': [1000, 1500, 1200, 1800],
        '成本': [600, 900, 700, 1000]
    })
    print(df_from_dict_pl)

    # 2. 从列表的列表创建
    print("\n2️⃣ 从列表创建DataFrame:")
    df_from_list_pl = pl.DataFrame(
        {
            '姓名': ['张三', '李四', '王五'],
            '年龄': [25, 30, 35],
            '城市': ['北京', '上海', '广州']
        },
        strict=False
    )
    print(df_from_list_pl)

    # 3. 从NumPy数组创建
    print("\n3️⃣ 从NumPy数组创建:")
    np_array_pl = np.random.rand(5, 3)
    df_from_numpy_pl = pl.DataFrame(np_array_pl, schema=['A', 'B', 'C'])
    print(df_from_numpy_pl)

    # 4. 从Pandas DataFrame创建
    print("\n4️⃣ 从Pandas DataFrame创建:")
    pandas_df_pl = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [4, 5, 6]
    })
    df_from_pandas_pl = pl.from_pandas(pandas_df_pl)
    print(df_from_pandas_pl)

    # 5. 读取CSV
    print("\n5️⃣ 读取CSV（示例）:")
    print("   pl.read_csv('data.csv')")
    print("   pl.scan_csv('data.csv')  # 惰性读取")

    return df_from_dict_pl, df_from_list_pl, df_from_numpy_pl, df_from_pandas_pl, np_array_pl, pandas_df_pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 👀 3. 数据查看和检查

    Polars提供了多种方法来查看和检查数据。
    """
    )
    return


@app.cell
def _(datetime, np, pl):
    print("=" * 60)
    print("👀 数据查看示例")
    print("=" * 60)

    # 创建示例数据
    np.random.seed(42)

    # 生成100天的日期
    dates_list = [datetime(2024, 1, 1) + pl.duration(days=i) for i in range(100)]

    sample_pl_df = pl.DataFrame({
        '日期': dates_list,
        '销售额': np.random.randint(1000, 10000, 100),
        '成本': np.random.randint(500, 5000, 100),
        '地区': np.random.choice(['北区', '南区', '东区', '西区'], 100),
        '产品': np.random.choice(['产品A', '产品B', '产品C'], 100)
    })

    # 1. 查看前几行
    print("\n1️⃣ 查看前5行 (head):")
    print(sample_pl_df.head())

    # 2. 查看后几行
    print("\n2️⃣ 查看后3行 (tail):")
    print(sample_pl_df.tail(3))

    # 3. 查看数据形状
    print("\n3️⃣ 数据形状:")
    print(f"   形状: {sample_pl_df.shape}")
    print(f"   行数: {sample_pl_df.height}")
    print(f"   列数: {sample_pl_df.width}")

    # 4. 查看列名和数据类型
    print("\n4️⃣ 列信息:")
    print(f"   列名: {sample_pl_df.columns}")
    print(f"   数据类型: {sample_pl_df.dtypes}")
    print(f"   Schema: {sample_pl_df.schema}")

    # 5. 描述性统计
    print("\n5️⃣ 描述性统计 (describe):")
    print(sample_pl_df.describe())

    # 6. 查看唯一值
    print("\n6️⃣ 地区的唯一值:")
    print(f"   唯一值: {sample_pl_df['地区'].unique().to_list()}")
    print(f"   唯一值数量: {sample_pl_df['地区'].n_unique()}")
    print(f"   值计数:\n{sample_pl_df['地区'].value_counts()}")

    return (sample_pl_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🎯 4. 数据选择和表达式API

    Polars的表达式API是其最强大的特性之一，使用`pl.col()`来引用列。

    ### 表达式的优势

    - **可组合**：可以链式调用多个操作
    - **并行化**：自动并行执行
    - **优化**：查询优化器自动优化
    - **类型安全**：编译时类型检查
    """
    )
    return


@app.cell
def _(pl, sample_pl_df):
    print("=" * 60)
    print("🎯 数据选择示例")
    print("=" * 60)

    # 1. 选择单列
    print("\n1️⃣ 选择单列:")
    print(sample_pl_df.select(pl.col('销售额')).head())

    # 2. 选择多列
    print("\n2️⃣ 选择多列:")
    print(sample_pl_df.select(['日期', '销售额', '地区']).head())

    # 3. 使用表达式选择
    print("\n3️⃣ 使用表达式选择:")
    print(sample_pl_df.select([
        pl.col('销售额'),
        pl.col('成本'),
        (pl.col('销售额') - pl.col('成本')).alias('利润')
    ]).head())

    # 4. 条件过滤
    print("\n4️⃣ 条件过滤（销售额>5000）:")
    high_sales_pl = sample_pl_df.filter(pl.col('销售额') > 5000)
    print(f"   符合条件的记录数: {high_sales_pl.height}")
    print(high_sales_pl.head())

    # 5. 多条件过滤
    print("\n5️⃣ 多条件过滤（销售额>5000 且 地区='北区'）:")
    complex_filter_pl = sample_pl_df.filter(
        (pl.col('销售额') > 5000) & (pl.col('地区') == '北区')
    )
    print(f"   符合条件的记录数: {complex_filter_pl.height}")
    print(complex_filter_pl.head())

    # 6. 使用is_in过滤
    print("\n6️⃣ 使用is_in过滤（产品为A或B）:")
    product_filter_pl = sample_pl_df.filter(
        pl.col('产品').is_in(['产品A', '产品B'])
    )
    print(f"   符合条件的记录数: {product_filter_pl.height}")

    # 7. 选择所有数值列
    print("\n7️⃣ 选择所有数值列:")
    numeric_cols = sample_pl_df.select(pl.col(pl.Int64, pl.Float64))
    print(numeric_cols.head())

    return complex_filter_pl, high_sales_pl, numeric_cols, product_filter_pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧹 5. 数据清洗

    Polars提供了高效的数据清洗方法。
    """
    )
    return


@app.cell
def _(pl):
    print("=" * 60)
    print("🧹 数据清洗示例")
    print("=" * 60)

    # 创建包含缺失值和重复值的数据
    dirty_pl_data = pl.DataFrame({
        '姓名': ['张三', '李四', '王五', '张三', '赵六', None, '钱七'],
        '年龄': [25, 30, None, 25, 28, 32, 35],
        '城市': ['北京', '上海', '广州', '北京', None, '杭州', '深圳'],
        '薪资': [15000, 18000, 16000, 15000, 17000, 19000, None]
    })

    print("\n原始数据（包含缺失值和重复值）:")
    print(dirty_pl_data)

    # 1. 检查缺失值
    print("\n1️⃣ 检查缺失值:")
    null_counts = dirty_pl_data.null_count()
    print(null_counts)

    # 2. 删除包含缺失值的行
    print("\n2️⃣ 删除包含缺失值的行:")
    cleaned_dropna_pl = dirty_pl_data.drop_nulls()
    print(cleaned_dropna_pl)

    # 3. 填充缺失值
    print("\n3️⃣ 填充缺失值:")
    filled_pl_data = dirty_pl_data.with_columns([
        pl.col('年龄').fill_null(pl.col('年龄').mean()),
        pl.col('城市').fill_null('未知'),
        pl.col('薪资').fill_null(pl.col('薪资').median()),
        pl.col('姓名').fill_null('匿名')
    ])
    print(filled_pl_data)

    # 4. 检查重复值
    print("\n4️⃣ 检查重复值:")
    is_duplicated = dirty_pl_data.is_duplicated()
    print(f"   重复行数: {is_duplicated.sum()}")

    # 5. 删除重复值
    print("\n5️⃣ 删除重复值:")
    deduped_pl_data = dirty_pl_data.unique()
    print(deduped_pl_data)

    # 6. 数据类型转换
    print("\n6️⃣ 数据类型转换:")
    type_converted_pl = filled_pl_data.with_columns([
        pl.col('年龄').cast(pl.Int32)
    ])
    print(f"   转换后的数据类型: {type_converted_pl.schema}")

    return cleaned_dropna_pl, deduped_pl_data, dirty_pl_data, filled_pl_data, is_duplicated, null_counts, type_converted_pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🔄 6. 数据转换

    Polars的表达式API使数据转换变得简单而高效。
    """
    )
    return


@app.cell
def _(pl, sample_pl_df):
    print("=" * 60)
    print("🔄 数据转换示例")
    print("=" * 60)

    # 1. 排序
    print("\n1️⃣ 按销售额降序排序:")
    sorted_pl_df = sample_pl_df.sort('销售额', descending=True)
    print(sorted_pl_df.head())

    # 2. 多列排序
    print("\n2️⃣ 按地区升序、销售额降序排序:")
    multi_sorted_pl = sample_pl_df.sort(['地区', '销售额'], descending=[False, True])
    print(multi_sorted_pl.head(10))

    # 3. 添加新列
    print("\n3️⃣ 添加利润列:")
    transform_pl_df = sample_pl_df.with_columns([
        (pl.col('销售额') - pl.col('成本')).alias('利润'),
        ((pl.col('销售额') - pl.col('成本')) / pl.col('销售额') * 100).round(2).alias('利润率')
    ])
    print(transform_pl_df.select(['日期', '销售额', '成本', '利润', '利润率']).head())

    # 4. 使用when-then-otherwise（类似SQL的CASE）
    print("\n4️⃣ 使用when-then-otherwise:")
    categorized_pl = sample_pl_df.with_columns([
        pl.when(pl.col('销售额') >= 7000)
          .then(pl.lit('高'))
          .when(pl.col('销售额') >= 4000)
          .then(pl.lit('中'))
          .otherwise(pl.lit('低'))
          .alias('销售等级')
    ])
    print(categorized_pl.select(['销售额', '销售等级']).head(10))

    # 5. 字符串操作
    print("\n5️⃣ 字符串操作:")
    string_ops_pl = sample_pl_df.with_columns([
        pl.col('地区').str.replace('区', '部').alias('部门'),
        pl.col('产品').str.to_uppercase().alias('产品大写')
    ])
    print(string_ops_pl.select(['地区', '部门', '产品', '产品大写']).head())

    # 6. 应用自定义函数
    print("\n6️⃣ 应用自定义函数:")
    def custom_func(x):
        return x * 1.1

    custom_applied = sample_pl_df.with_columns([
        pl.col('销售额').map_elements(custom_func, return_dtype=pl.Float64).alias('调整后销售额')
    ])
    print(custom_applied.select(['销售额', '调整后销售额']).head())

    return categorized_pl, custom_applied, custom_func, multi_sorted_pl, sorted_pl_df, string_ops_pl, transform_pl_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📊 7. 数据聚合和分组

    Polars的分组聚合操作非常高效。
    """
    )
    return


@app.cell
def _(pl, sample_pl_df):
    print("=" * 60)
    print("📊 数据聚合和分组示例")
    print("=" * 60)

    # 1. 按单列分组聚合
    print("\n1️⃣ 按地区分组，计算平均销售额:")
    region_avg_pl = sample_pl_df.group_by('地区').agg([
        pl.col('销售额').mean().round(2).alias('平均销售额')
    ])
    print(region_avg_pl)

    # 2. 按多列分组
    print("\n2️⃣ 按地区和产品分组，计算总销售额:")
    multi_group_pl = sample_pl_df.group_by(['地区', '产品']).agg([
        pl.col('销售额').sum().alias('总销售额')
    ])
    print(multi_group_pl)

    # 3. 多种聚合函数
    print("\n3️⃣ 按地区分组，应用多种聚合函数:")
    agg_result_pl = sample_pl_df.group_by('地区').agg([
        pl.col('销售额').sum().alias('总销售额'),
        pl.col('销售额').mean().round(2).alias('平均销售额'),
        pl.col('销售额').min().alias('最小销售额'),
        pl.col('销售额').max().alias('最大销售额'),
        pl.col('销售额').count().alias('记录数')
    ])
    print(agg_result_pl)

    # 4. 对不同列应用不同聚合函数
    print("\n4️⃣ 对不同列应用不同聚合函数:")
    complex_agg_pl = sample_pl_df.group_by('地区').agg([
        pl.col('销售额').sum().alias('总销售额'),
        pl.col('销售额').mean().round(2).alias('平均销售额'),
        pl.col('成本').sum().alias('总成本'),
        pl.col('成本').mean().round(2).alias('平均成本')
    ])
    print(complex_agg_pl)

    # 5. 使用over进行窗口函数
    print("\n5️⃣ 使用over添加组内平均值:")
    window_result_pl = sample_pl_df.with_columns([
        pl.col('销售额').mean().over('地区').round(2).alias('地区平均销售额')
    ])
    print(window_result_pl.select(['地区', '销售额', '地区平均销售额']).head(10))

    # 6. 透视表
    print("\n6️⃣ 创建透视表:")
    pivot_result_pl = sample_pl_df.pivot(
        values='销售额',
        index='地区',
        columns='产品',
        aggregate_function='mean'
    )
    print(pivot_result_pl)

    return agg_result_pl, complex_agg_pl, multi_group_pl, pivot_result_pl, region_avg_pl, window_result_pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🔗 8. 数据合并和连接

    Polars提供了高效的数据合并操作。
    """
    )
    return


@app.cell
def _(pl):
    print("=" * 60)
    print("🔗 数据合并和连接示例")
    print("=" * 60)

    # 创建示例数据
    employees_pl = pl.DataFrame({
        '员工ID': [1, 2, 3, 4],
        '姓名': ['张三', '李四', '王五', '赵六'],
        '部门ID': [101, 102, 101, 103]
    })

    departments_pl = pl.DataFrame({
        '部门ID': [101, 102, 103, 104],
        '部门名称': ['技术部', '销售部', '人事部', '财务部']
    })

    salaries_pl = pl.DataFrame({
        '员工ID': [1, 2, 3, 5],
        '薪资': [15000, 18000, 16000, 20000]
    })

    print("\n员工表:")
    print(employees_pl)
    print("\n部门表:")
    print(departments_pl)
    print("\n薪资表:")
    print(salaries_pl)

    # 1. 内连接（inner join）
    print("\n1️⃣ 内连接（员工和部门）:")
    inner_join_pl = employees_pl.join(departments_pl, on='部门ID', how='inner')
    print(inner_join_pl)

    # 2. 左连接（left join）
    print("\n2️⃣ 左连接（员工和薪资）:")
    left_join_pl = employees_pl.join(salaries_pl, on='员工ID', how='left')
    print(left_join_pl)

    # 3. 外连接（outer join）
    print("\n3️⃣ 外连接（员工和薪资）:")
    outer_join_pl = employees_pl.join(salaries_pl, on='员工ID', how='outer')
    print(outer_join_pl)

    # 4. 多表连接
    print("\n4️⃣ 多表连接:")
    full_info_pl = (employees_pl
                    .join(departments_pl, on='部门ID')
                    .join(salaries_pl, on='员工ID', how='left'))
    print(full_info_pl)

    # 5. 垂直拼接
    print("\n5️⃣ 垂直拼接（concat）:")
    df1_pl = pl.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2_pl = pl.DataFrame({'A': [5, 6], 'B': [7, 8]})
    concat_result_pl = pl.concat([df1_pl, df2_pl])
    print(concat_result_pl)

    # 6. 水平拼接
    print("\n6️⃣ 水平拼接:")
    df3_pl = pl.DataFrame({'C': [9, 10]})
    hconcat_result_pl = pl.concat([df1_pl, df3_pl], how='horizontal')
    print(hconcat_result_pl)

    return concat_result_pl, departments_pl, df1_pl, df2_pl, df3_pl, employees_pl, full_info_pl, hconcat_result_pl, inner_join_pl, left_join_pl, outer_join_pl, salaries_pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ⚡ 9. 惰性求值和查询优化

    LazyFrame是Polars的杀手级特性，可以自动优化查询。
    """
    )
    return


@app.cell
def _(pl, sample_pl_df):
    print("=" * 60)
    print("⚡ 惰性求值示例")
    print("=" * 60)

    # 1. 创建LazyFrame
    print("\n1️⃣ 创建LazyFrame:")
    lazy_df = sample_pl_df.lazy()
    print(f"   类型: {type(lazy_df)}")

    # 2. 构建查询（不会立即执行）
    print("\n2️⃣ 构建查询（不会立即执行）:")
    query = (lazy_df
             .filter(pl.col('销售额') > 5000)
             .group_by('地区')
             .agg([
                 pl.col('销售额').sum().alias('总销售额'),
                 pl.col('销售额').mean().alias('平均销售额')
             ])
             .sort('总销售额', descending=True))

    print("   查询已构建，但未执行")
    print(f"   查询类型: {type(query)}")

    # 3. 查看查询计划
    print("\n3️⃣ 查看优化后的查询计划:")
    print(query.explain())

    # 4. 执行查询
    print("\n4️⃣ 执行查询:")
    result_lazy = query.collect()
    print(result_lazy)

    # 5. 流式处理大文件
    print("\n5️⃣ 流式处理（示例）:")
    print("   # 扫描CSV文件（不加载到内存）")
    print("   lazy_csv = pl.scan_csv('large_file.csv')")
    print("   ")
    print("   # 构建查询")
    print("   result = (lazy_csv")
    print("       .filter(pl.col('amount') > 1000)")
    print("       .group_by('category')")
    print("       .agg(pl.col('amount').sum())")
    print("       .collect()  # 只在这里才执行")
    print("   )")

    # 6. 查询优化示例
    print("\n6️⃣ 查询优化示例:")
    print("   Polars会自动优化以下操作：")
    print("   - 谓词下推（Predicate Pushdown）")
    print("   - 投影下推（Projection Pushdown）")
    print("   - 公共子表达式消除")
    print("   - 并行执行")

    return lazy_df, query, result_lazy


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🏎️ 10. 性能对比：Polars vs Pandas

    让我们通过实际测试来对比Polars和Pandas的性能。
    """
    )
    return


@app.cell
def _(np, pd, pl, time):
    print("=" * 60)
    print("🏎️ 性能对比：Polars vs Pandas")
    print("=" * 60)

    # 创建测试数据
    n_rows = 1_000_000
    print(f"\n测试数据规模: {n_rows:,} 行")

    np.random.seed(42)
    test_data = {
        'id': range(n_rows),
        'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows),
        'value1': np.random.randn(n_rows),
        'value2': np.random.randn(n_rows),
        'value3': np.random.randint(0, 100, n_rows)
    }

    # 创建Pandas DataFrame
    print("\n创建Pandas DataFrame...")
    start = time.time()
    df_pandas = pd.DataFrame(test_data)
    pandas_create_time = time.time() - start
    print(f"   Pandas创建时间: {pandas_create_time:.4f}秒")

    # 创建Polars DataFrame
    print("\n创建Polars DataFrame...")
    start = time.time()
    df_polars = pl.DataFrame(test_data)
    polars_create_time = time.time() - start
    print(f"   Polars创建时间: {polars_create_time:.4f}秒")
    print(f"   ⚡ Polars快 {pandas_create_time/polars_create_time:.2f}x")

    # 测试1: 过滤操作
    print("\n" + "=" * 60)
    print("测试1: 过滤操作 (value1 > 0)")
    print("=" * 60)

    start = time.time()
    pandas_filtered = df_pandas[df_pandas['value1'] > 0]
    pandas_filter_time = time.time() - start
    print(f"   Pandas: {pandas_filter_time:.4f}秒")

    start = time.time()
    polars_filtered = df_polars.filter(pl.col('value1') > 0)
    polars_filter_time = time.time() - start
    print(f"   Polars: {polars_filter_time:.4f}秒")
    print(f"   ⚡ Polars快 {pandas_filter_time/polars_filter_time:.2f}x")

    # 测试2: 分组聚合
    print("\n" + "=" * 60)
    print("测试2: 分组聚合")
    print("=" * 60)

    start = time.time()
    pandas_grouped = df_pandas.groupby('category').agg({
        'value1': 'mean',
        'value2': 'sum',
        'value3': 'max'
    })
    pandas_group_time = time.time() - start
    print(f"   Pandas: {pandas_group_time:.4f}秒")

    start = time.time()
    polars_grouped = df_polars.group_by('category').agg([
        pl.col('value1').mean(),
        pl.col('value2').sum(),
        pl.col('value3').max()
    ])
    polars_group_time = time.time() - start
    print(f"   Polars: {polars_group_time:.4f}秒")
    print(f"   ⚡ Polars快 {pandas_group_time/polars_group_time:.2f}x")

    # 测试3: 排序
    print("\n" + "=" * 60)
    print("测试3: 排序")
    print("=" * 60)

    start = time.time()
    pandas_sorted = df_pandas.sort_values('value1')
    pandas_sort_time = time.time() - start
    print(f"   Pandas: {pandas_sort_time:.4f}秒")

    start = time.time()
    polars_sorted = df_polars.sort('value1')
    polars_sort_time = time.time() - start
    print(f"   Polars: {polars_sort_time:.4f}秒")
    print(f"   ⚡ Polars快 {pandas_sort_time/polars_sort_time:.2f}x")

    # 性能总结
    print("\n" + "=" * 60)
    print("📊 性能总结")
    print("=" * 60)
    print(f"   创建: Polars快 {pandas_create_time/polars_create_time:.2f}x")
    print(f"   过滤: Polars快 {pandas_filter_time/polars_filter_time:.2f}x")
    print(f"   分组: Polars快 {pandas_group_time/polars_group_time:.2f}x")
    print(f"   排序: Polars快 {pandas_sort_time/polars_sort_time:.2f}x")

    return df_pandas, df_polars, n_rows, pandas_create_time, pandas_filter_time, pandas_filtered, pandas_group_time, pandas_grouped, pandas_sort_time, pandas_sorted, polars_create_time, polars_filter_time, polars_filtered, polars_group_time, polars_grouped, polars_sort_time, polars_sorted, start, test_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🔄 11. 从Pandas迁移到Polars

    ### 常用操作对照表

    | 操作 | Pandas | Polars |
    |------|--------|--------|
    | **创建DataFrame** | `pd.DataFrame(data)` | `pl.DataFrame(data)` |
    | **读取CSV** | `pd.read_csv('file.csv')` | `pl.read_csv('file.csv')` |
    | **查看前几行** | `df.head()` | `df.head()` |
    | **选择列** | `df['col']` | `df['col']` 或 `df.select('col')` |
    | **选择多列** | `df[['a', 'b']]` | `df.select(['a', 'b'])` |
    | **过滤** | `df[df['col'] > 5]` | `df.filter(pl.col('col') > 5)` |
    | **添加列** | `df['new'] = df['a'] + df['b']` | `df.with_columns((pl.col('a') + pl.col('b')).alias('new'))` |
    | **排序** | `df.sort_values('col')` | `df.sort('col')` |
    | **分组聚合** | `df.groupby('col').agg({'a': 'sum'})` | `df.group_by('col').agg(pl.col('a').sum())` |
    | **连接** | `pd.merge(df1, df2, on='key')` | `df1.join(df2, on='key')` |
    | **缺失值** | `df.fillna(0)` | `df.fill_null(0)` |
    | **重复值** | `df.drop_duplicates()` | `df.unique()` |
    | **应用函数** | `df['col'].apply(func)` | `df.select(pl.col('col').map_elements(func))` |

    ### 关键差异

    #### 1. 表达式API vs 方法链

    **Pandas**:
    ```python
    df['new_col'] = df['a'] + df['b']
    df = df[df['new_col'] > 10]
    ```

    **Polars**:
    ```python
    df = df.with_columns([
        (pl.col('a') + pl.col('b')).alias('new_col')
    ]).filter(pl.col('new_col') > 10)
    ```

    #### 2. 惰性求值

    **Pandas**: 总是即时执行
    ```python
    result = df.groupby('category').sum()  # 立即执行
    ```

    **Polars**: 可以选择惰性执行
    ```python
    # 即时执行
    result = df.group_by('category').sum()

    # 惰性执行（推荐用于大数据）
    result = df.lazy().group_by('category').sum().collect()
    ```

    #### 3. 列引用方式

    **Pandas**: 直接使用列名字符串
    ```python
    df['column_name']
    ```

    **Polars**: 使用`pl.col()`表达式
    ```python
    df.select(pl.col('column_name'))
    ```

    #### 4. 不可变性

    **Pandas**: 默认可变，支持`inplace=True`
    ```python
    df.fillna(0, inplace=True)
    ```

    **Polars**: 不可变，总是返回新对象
    ```python
    df = df.fill_null(0)
    ```

    ### 迁移建议

    1. **逐步迁移**：不需要一次性全部迁移，可以混用
    2. **使用惰性求值**：对于大数据集，使用LazyFrame
    3. **学习表达式API**：这是Polars的核心优势
    4. **利用类型系统**：Polars的严格类型可以提前发现错误
    5. **性能测试**：在实际场景中测试性能提升
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🚀 12. 完整实战案例

    让我们通过一个电商数据分析案例来综合运用Polars的各种功能。
    """
    )
    return


@app.cell
def _(datetime, np, pl, timedelta):
    print("=" * 60)
    print("🚀 完整实战案例：电商销售数据分析（Polars版）")
    print("=" * 60)

    # 1. 创建模拟数据
    print("\n📊 步骤1：创建模拟电商销售数据")
    np.random.seed(42)
    n_records_ecom = 10000

    # 生成日期列表
    dates_ecom = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_records_ecom)]

    ecommerce_pl_data = pl.DataFrame({
        '订单ID': range(1, n_records_ecom + 1),
        '日期': dates_ecom,
        '用户ID': np.random.randint(1000, 2000, n_records_ecom),
        '产品类别': np.random.choice(['电子产品', '服装', '食品', '图书', '家居'], n_records_ecom),
        '数量': np.random.randint(1, 10, n_records_ecom),
        '单价': np.random.uniform(10, 1000, n_records_ecom).round(2),
        '地区': np.random.choice(['华北', '华东', '华南', '华中', '西南', '东北'], n_records_ecom),
        '支付方式': np.random.choice(['支付宝', '微信', '信用卡', '货到付款'], n_records_ecom)
    })

    print(f"数据形状: {ecommerce_pl_data.shape}")
    print("\n前5条记录:")
    print(ecommerce_pl_data.head())

    # 2. 数据清洗和特征工程（使用表达式API）
    print("\n🔧 步骤2：数据清洗和特征工程")

    ecommerce_pl_data = ecommerce_pl_data.with_columns([
        # 计算总金额
        (pl.col('数量') * pl.col('单价')).round(2).alias('总金额'),
        # 提取时间特征
        pl.col('日期').dt.year().alias('年'),
        pl.col('日期').dt.month().alias('月'),
        pl.col('日期').dt.day().alias('日'),
        pl.col('日期').dt.hour().alias('小时'),
        pl.col('日期').dt.weekday().alias('星期'),
        # 是否周末
        (pl.col('日期').dt.weekday().is_in([5, 6])).alias('是否周末')
    ])

    print("新增特征:")
    print(ecommerce_pl_data.select(['日期', '总金额', '年', '月', '日', '小时', '星期', '是否周末']).head())

    return ecommerce_pl_data, n_records_ecom


@app.cell
def _(ecommerce_pl_data, pl):
    print("=" * 60)
    print("📈 步骤3：探索性数据分析（使用Polars表达式）")
    print("=" * 60)

    # 1. 按产品类别分析
    print("\n1️⃣ 按产品类别分析:")
    category_analysis_pl = ecommerce_pl_data.group_by('产品类别').agg([
        pl.col('订单ID').count().alias('订单数'),
        pl.col('总金额').sum().round(2).alias('总销售额'),
        pl.col('总金额').mean().round(2).alias('平均订单金额'),
        pl.col('总金额').max().alias('最大订单金额'),
        pl.col('数量').sum().alias('总销量')
    ]).sort('总销售额', descending=True)
    print(category_analysis_pl)

    # 2. 按地区分析
    print("\n2️⃣ 按地区分析:")
    region_analysis_pl = ecommerce_pl_data.group_by('地区').agg([
        pl.col('订单ID').count().alias('订单数'),
        pl.col('总金额').sum().round(2).alias('总销售额')
    ]).sort('总销售额', descending=True)
    print(region_analysis_pl)

    # 3. 按支付方式分析
    print("\n3️⃣ 按支付方式分析:")
    payment_analysis_pl = ecommerce_pl_data.group_by('支付方式').agg([
        pl.col('订单ID').count().alias('订单数')
    ]).with_columns([
        (pl.col('订单数') / pl.col('订单数').sum() * 100).round(2).alias('占比(%)')
    ]).sort('订单数', descending=True)
    print(payment_analysis_pl)

    # 4. 时间趋势分析（按日期）
    print("\n4️⃣ 按日期分析销售趋势:")
    daily_sales_pl = ecommerce_pl_data.group_by(pl.col('日期').dt.date()).agg([
        pl.col('订单ID').count().alias('订单数'),
        pl.col('总金额').sum().round(2).alias('总销售额')
    ]).sort('日期')
    print(daily_sales_pl.head(10))

    # 5. 按小时分析
    print("\n5️⃣ 按小时分析订单分布:")
    hourly_orders_pl = ecommerce_pl_data.group_by('小时').agg([
        pl.col('订单ID').count().alias('订单数')
    ]).sort('小时')
    print(hourly_orders_pl)

    # 6. 周末vs工作日
    print("\n6️⃣ 周末vs工作日对比:")
    weekend_analysis_pl = ecommerce_pl_data.group_by('是否周末').agg([
        pl.col('订单ID').count().alias('订单数'),
        pl.col('总金额').sum().round(2).alias('总销售额'),
        pl.col('总金额').mean().round(2).alias('平均订单金额')
    ])
    print(weekend_analysis_pl)

    return category_analysis_pl, daily_sales_pl, hourly_orders_pl, payment_analysis_pl, region_analysis_pl, weekend_analysis_pl


@app.cell
def _(ecommerce_pl_data, pl):
    print("=" * 60)
    print("🎯 步骤4：高级分析（使用LazyFrame优化）")
    print("=" * 60)

    # 使用LazyFrame进行复杂查询
    lazy_ecom = ecommerce_pl_data.lazy()

    # 1. 用户购买行为分析
    print("\n1️⃣ 用户购买行为分析:")
    user_behavior_pl = (lazy_ecom
        .group_by('用户ID')
        .agg([
            pl.col('订单ID').count().alias('购买次数'),
            pl.col('总金额').sum().round(2).alias('总消费金额'),
            pl.col('产品类别').n_unique().alias('购买类别数')
        ])
        .with_columns([
            (pl.col('总消费金额') / pl.col('购买次数')).round(2).alias('平均订单金额')
        ])
        .collect()
    )

    print("用户行为统计:")
    print(user_behavior_pl.describe())

    print("\n消费金额TOP10用户:")
    print(user_behavior_pl.sort('总消费金额', descending=True).head(10))

    # 2. 产品类别组合分析
    print("\n2️⃣ 产品类别和地区组合分析:")
    category_region_pl = (lazy_ecom
        .group_by(['产品类别', '地区'])
        .agg([
            pl.col('总金额').sum().round(2).alias('总销售额'),
            pl.col('订单ID').count().alias('订单数')
        ])
        .sort('总销售额', descending=True)
        .collect()
    )
    print(category_region_pl.head(10))

    # 3. 客单价分析
    print("\n3️⃣ 客单价分布分析:")
    order_value_stats = ecommerce_pl_data.group_by('订单ID').agg([
        pl.col('总金额').sum().alias('订单总额')
    ])

    print(f"平均客单价: {order_value_stats['订单总额'].mean():.2f}")
    print(f"中位数客单价: {order_value_stats['订单总额'].median():.2f}")
    print(f"最高客单价: {order_value_stats['订单总额'].max():.2f}")
    print(f"最低客单价: {order_value_stats['订单总额'].min():.2f}")

    # 4. 使用窗口函数计算排名
    print("\n4️⃣ 各地区销售额TOP3产品:")
    top_products_pl = (lazy_ecom
        .group_by(['地区', '产品类别'])
        .agg([
            pl.col('总金额').sum().alias('销售额')
        ])
        .with_columns([
            pl.col('销售额').rank(method='dense', descending=True).over('地区').alias('排名')
        ])
        .filter(pl.col('排名') <= 3)
        .sort(['地区', '排名'])
        .collect()
    )
    print(top_products_pl)

    return category_region_pl, lazy_ecom, order_value_stats, top_products_pl, user_behavior_pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📚 13. Polars最佳实践和总结

    ### 最佳实践

    #### 1. 何时使用Polars

    ✅ **适合使用Polars的场景**：
    - 处理大数据集（GB级别）
    - 需要高性能计算
    - 复杂的数据转换和聚合
    - 批处理和ETL任务
    - 需要并行处理

    ⚠️ **可能不适合的场景**：
    - 小数据集（<1MB）
    - 需要丰富的生态系统（如可视化库）
    - 团队不熟悉Polars
    - 需要与现有Pandas代码深度集成

    #### 2. 性能优化技巧

    1. **使用LazyFrame**：对于大数据集和复杂查询
    2. **避免collect()过早**：尽可能延迟执行
    3. **使用表达式API**：比循环快得多
    4. **利用并行处理**：Polars自动并行化
    5. **选择合适的数据类型**：减少内存占用
    6. **使用scan_csv而不是read_csv**：对于大文件

    #### 3. 代码风格建议

    **推荐**：使用链式调用和表达式
    ```python
    result = (df
        .filter(pl.col('value') > 0)
        .group_by('category')
        .agg([
            pl.col('amount').sum(),
            pl.col('count').mean()
        ])
        .sort('amount', descending=True)
    )
    ```

    **避免**：逐步赋值
    ```python
    df = df.filter(pl.col('value') > 0)
    df = df.group_by('category')
    # ...
    ```

    ### 常用方法速查表

    #### 📊 数据创建和IO

    | 方法 | 说明 | 示例 |
    |------|------|------|
    | `pl.DataFrame()` | 创建DataFrame | `pl.DataFrame({'A': [1, 2, 3]})` |
    | `pl.LazyFrame()` | 创建LazyFrame | `pl.LazyFrame({'A': [1, 2, 3]})` |
    | `pl.read_csv()` | 读取CSV（即时） | `pl.read_csv('data.csv')` |
    | `pl.scan_csv()` | 扫描CSV（惰性） | `pl.scan_csv('data.csv')` |
    | `pl.read_parquet()` | 读取Parquet | `pl.read_parquet('data.parquet')` |
    | `pl.from_pandas()` | 从Pandas转换 | `pl.from_pandas(pandas_df)` |
    | `df.write_csv()` | 写入CSV | `df.write_csv('output.csv')` |
    | `df.write_parquet()` | 写入Parquet | `df.write_parquet('output.parquet')` |

    #### 👀 数据查看

    | 方法 | 说明 | 示例 |
    |------|------|------|
    | `df.head(n)` | 查看前n行 | `df.head(10)` |
    | `df.tail(n)` | 查看后n行 | `df.tail(10)` |
    | `df.sample(n)` | 随机抽样 | `df.sample(5)` |
    | `df.describe()` | 描述性统计 | `df.describe()` |
    | `df.shape` | 数据形状 | `df.shape` |
    | `df.height` | 行数 | `df.height` |
    | `df.width` | 列数 | `df.width` |
    | `df.columns` | 列名 | `df.columns` |
    | `df.dtypes` | 数据类型 | `df.dtypes` |
    | `df.schema` | Schema信息 | `df.schema` |

    #### 🎯 数据选择

    | 方法 | 说明 | 示例 |
    |------|------|------|
    | `df.select()` | 选择列 | `df.select(['a', 'b'])` |
    | `df.select(pl.col())` | 表达式选择 | `df.select(pl.col('a'))` |
    | `df.filter()` | 过滤行 | `df.filter(pl.col('a') > 5)` |
    | `df['col']` | 获取列 | `df['column_name']` |
    | `df[0]` | 获取行 | `df[0]` |
    | `df[0:5]` | 切片 | `df[0:5]` |
    | `pl.col()` | 列引用 | `pl.col('column_name')` |
    | `pl.all()` | 所有列 | `df.select(pl.all())` |
    | `pl.exclude()` | 排除列 | `df.select(pl.exclude('a'))` |

    #### 🧹 数据清洗

    | 方法 | 说明 | 示例 |
    |------|------|------|
    | `df.null_count()` | 统计缺失值 | `df.null_count()` |
    | `df.drop_nulls()` | 删除缺失值 | `df.drop_nulls()` |
    | `df.fill_null()` | 填充缺失值 | `df.fill_null(0)` |
    | `df.fill_nan()` | 填充NaN | `df.fill_nan(0)` |
    | `df.is_duplicated()` | 检查重复 | `df.is_duplicated()` |
    | `df.unique()` | 删除重复 | `df.unique()` |
    | `df.drop()` | 删除列 | `df.drop('column')` |
    | `pl.col().cast()` | 类型转换 | `pl.col('a').cast(pl.Int32)` |

    #### 🔄 数据转换

    | 方法 | 说明 | 示例 |
    |------|------|------|
    | `df.with_columns()` | 添加/修改列 | `df.with_columns(pl.col('a') + 1)` |
    | `df.sort()` | 排序 | `df.sort('column')` |
    | `df.rename()` | 重命名 | `df.rename({'old': 'new'})` |
    | `pl.when().then().otherwise()` | 条件表达式 | `pl.when(cond).then(1).otherwise(0)` |
    | `pl.col().alias()` | 列别名 | `pl.col('a').alias('new_name')` |
    | `pl.col().map_elements()` | 应用函数 | `pl.col('a').map_elements(func)` |
    | `df.explode()` | 展开列表 | `df.explode('list_column')` |

    #### 📊 数据聚合

    | 方法 | 说明 | 示例 |
    |------|------|------|
    | `df.group_by()` | 分组 | `df.group_by('category')` |
    | `df.agg()` | 聚合 | `df.agg(pl.col('a').sum())` |
    | `pl.col().sum()` | 求和 | `pl.col('a').sum()` |
    | `pl.col().mean()` | 平均值 | `pl.col('a').mean()` |
    | `pl.col().median()` | 中位数 | `pl.col('a').median()` |
    | `pl.col().min()` / `.max()` | 最小/最大值 | `pl.col('a').max()` |
    | `pl.col().count()` | 计数 | `pl.col('a').count()` |
    | `pl.col().n_unique()` | 唯一值数 | `pl.col('a').n_unique()` |
    | `pl.col().over()` | 窗口函数 | `pl.col('a').sum().over('group')` |
    | `df.pivot()` | 透视 | `df.pivot(values='v', index='i', columns='c')` |

    #### 🔗 数据合并

    | 方法 | 说明 | 示例 |
    |------|------|------|
    | `df.join()` | 连接 | `df1.join(df2, on='key')` |
    | `pl.concat()` | 拼接 | `pl.concat([df1, df2])` |
    | `df.hstack()` | 水平堆叠 | `df1.hstack(df2)` |
    | `df.vstack()` | 垂直堆叠 | `df1.vstack(df2)` |

    #### ⚡ 惰性求值

    | 方法 | 说明 | 示例 |
    |------|------|------|
    | `df.lazy()` | 转为LazyFrame | `df.lazy()` |
    | `lf.collect()` | 执行查询 | `lf.collect()` |
    | `lf.explain()` | 查看查询计划 | `lf.explain()` |
    | `pl.scan_csv()` | 惰性读取CSV | `pl.scan_csv('file.csv')` |

    ### 学习资源

    - 📖 [Polars官方文档](https://pola-rs.github.io/polars/)
    - 📚 [Polars用户指南](https://pola-rs.github.io/polars-book/)
    - 💡 [Polars GitHub](https://github.com/pola-rs/polars)
    - 🎓 [从Pandas到Polars](https://pola-rs.github.io/polars-book/user-guide/migration/pandas/)

    ### 总结

    **Polars的核心优势**：
    - ⚡ **性能**：比Pandas快10-100倍
    - 🧠 **内存效率**：更低的内存占用
    - 🔄 **惰性求值**：自动查询优化
    - 🎯 **表达式API**：强大而优雅
    - 🔗 **并行处理**：自动利用多核

    **何时选择Polars**：
    - 处理大数据集
    - 需要高性能
    - 复杂的数据转换
    - ETL和批处理任务

    **何时继续使用Pandas**：
    - 小数据集
    - 需要丰富的生态系统
    - 团队熟悉度
    - 与现有代码集成

    Polars是数据处理的未来，值得学习和使用！⚡🐻‍❄️
    """
    )
    return


if __name__ == "__main__":
    app.run()

