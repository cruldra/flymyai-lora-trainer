import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 🐼 Pandas完全指南

    ## 📚 什么是Pandas？

    **Pandas**是Python中最流行的数据分析和处理库，提供了高性能、易用的数据结构和数据分析工具。

    ### 核心特点

    - **强大的数据结构**：Series（一维）和DataFrame（二维）
    - **灵活的数据操作**：索引、切片、过滤、分组、聚合
    - **数据清洗**：处理缺失值、重复值、异常值
    - **数据转换**：合并、连接、重塑、透视
    - **时间序列**：强大的日期时间处理能力
    - **IO工具**：读写CSV、Excel、SQL、JSON等多种格式
    - **高性能**：基于NumPy构建，运算速度快

    ### 主要应用场景

    - 📊 数据清洗和预处理
    - 📈 探索性数据分析（EDA）
    - 🔄 数据转换和特征工程
    - 📉 时间序列分析
    - 📋 报表生成和数据可视化
    - 🤖 机器学习数据准备

    ### 本指南内容

    本笔记本将全面介绍Pandas的核心概念和常用API，包括：

    1. **数据结构**：Series和DataFrame
    2. **数据创建**：从各种数据源创建
    3. **数据查看**：查看和检查数据
    4. **数据选择**：索引、切片、过滤
    5. **数据清洗**：处理缺失值和重复值
    6. **数据转换**：排序、映射、应用函数
    7. **数据聚合**：分组和聚合操作
    8. **数据合并**：合并、连接、拼接
    9. **数据重塑**：透视、堆叠、融合
    10. **时间序列**：日期时间处理
    11. **数据IO**：读写各种格式
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
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import warnings
    warnings.filterwarnings('ignore')

    print(f"✅ Pandas版本: {pd.__version__}")
    print(f"✅ NumPy版本: {np.__version__}")
    return datetime, np, pd, timedelta, warnings


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📊 1. 核心数据结构

    Pandas有两个主要的数据结构：

    ### Series（一维数据）

    **Series**是带标签的一维数组，可以存储任何数据类型。

    **特点**：
    - 类似于Python的列表或NumPy数组
    - 每个元素都有一个标签（索引）
    - 可以通过标签或位置访问元素
    """
    )
    return


@app.cell
def _(np, pd):
    print("=" * 60)
    print("📊 Series示例")
    print("=" * 60)

    # 1. 从列表创建Series
    series_from_list = pd.Series([10, 20, 30, 40, 50])
    print("\n1️⃣ 从列表创建Series:")
    print(series_from_list)

    # 2. 带自定义索引的Series
    series_with_index = pd.Series([10, 20, 30, 40, 50], 
                                   index=['a', 'b', 'c', 'd', 'e'])
    print("\n2️⃣ 带自定义索引的Series:")
    print(series_with_index)

    # 3. 从字典创建Series
    series_from_dict = pd.Series({
        '北京': 2154,
        '上海': 2428,
        '广州': 1868,
        '深圳': 1756
    })
    print("\n3️⃣ 从字典创建Series（城市人口，万人）:")
    print(series_from_dict)

    # 4. Series的基本属性
    print("\n4️⃣ Series的基本属性:")
    print(f"   数据类型: {series_from_dict.dtype}")
    print(f"   形状: {series_from_dict.shape}")
    print(f"   大小: {series_from_dict.size}")
    print(f"   索引: {series_from_dict.index.tolist()}")
    print(f"   值: {series_from_dict.values}")

    # 5. Series的基本操作
    print("\n5️⃣ Series的基本操作:")
    print(f"   最大值: {series_from_dict.max()}")
    print(f"   最小值: {series_from_dict.min()}")
    print(f"   平均值: {series_from_dict.mean():.2f}")
    print(f"   总和: {series_from_dict.sum()}")

    return series_from_dict, series_from_list, series_with_index


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### DataFrame（二维数据）

    **DataFrame**是带标签的二维表格数据结构，类似于Excel表格或SQL表。

    **特点**：
    - 每列可以是不同的数据类型
    - 有行索引和列索引
    - 可以看作是Series的字典
    - 是Pandas中最常用的数据结构
    """
    )
    return


@app.cell
def _(np, pd):
    print("=" * 60)
    print("📋 DataFrame示例")
    print("=" * 60)

    # 1. 从字典创建DataFrame
    data_dict = {
        '姓名': ['张三', '李四', '王五', '赵六', '钱七'],
        '年龄': [25, 30, 35, 28, 32],
        '城市': ['北京', '上海', '广州', '深圳', '杭州'],
        '薪资': [15000, 18000, 16000, 17000, 19000]
    }
    df_from_dict = pd.DataFrame(data_dict)
    print("\n1️⃣ 从字典创建DataFrame:")
    print(df_from_dict)

    # 2. 从列表的列表创建DataFrame
    data_list = [
        ['张三', 25, '北京', 15000],
        ['李四', 30, '上海', 18000],
        ['王五', 35, '广州', 16000]
    ]
    df_from_list = pd.DataFrame(data_list, 
                                 columns=['姓名', '年龄', '城市', '薪资'])
    print("\n2️⃣ 从列表创建DataFrame:")
    print(df_from_list)

    # 3. DataFrame的基本属性
    print("\n3️⃣ DataFrame的基本属性:")
    print(f"   形状: {df_from_dict.shape}")
    print(f"   行数: {len(df_from_dict)}")
    print(f"   列数: {len(df_from_dict.columns)}")
    print(f"   列名: {df_from_dict.columns.tolist()}")
    print(f"   索引: {df_from_dict.index.tolist()}")
    print(f"   数据类型:\n{df_from_dict.dtypes}")

    # 4. DataFrame的基本信息
    print("\n4️⃣ DataFrame的基本信息:")
    print(df_from_dict.info())

    return data_dict, data_list, df_from_dict, df_from_list


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 👀 2. 数据查看和检查

    Pandas提供了多种方法来快速查看和检查数据。
    """
    )
    return


@app.cell
def _(np, pd):
    print("=" * 60)
    print("👀 数据查看示例")
    print("=" * 60)

    # 创建示例数据
    np.random.seed(42)
    sample_df = pd.DataFrame({
        '日期': pd.date_range('2024-01-01', periods=100),
        '销售额': np.random.randint(1000, 10000, 100),
        '成本': np.random.randint(500, 5000, 100),
        '地区': np.random.choice(['北区', '南区', '东区', '西区'], 100),
        '产品': np.random.choice(['产品A', '产品B', '产品C'], 100)
    })

    # 1. 查看前几行
    print("\n1️⃣ 查看前5行 (head):")
    print(sample_df.head())

    # 2. 查看后几行
    print("\n2️⃣ 查看后3行 (tail):")
    print(sample_df.tail(3))

    # 3. 随机抽样
    print("\n3️⃣ 随机抽样3行 (sample):")
    print(sample_df.sample(3))

    # 4. 描述性统计
    print("\n4️⃣ 描述性统计 (describe):")
    print(sample_df.describe())

    # 5. 查看数据类型
    print("\n5️⃣ 数据类型 (dtypes):")
    print(sample_df.dtypes)

    # 6. 查看唯一值
    print("\n6️⃣ 地区的唯一值:")
    print(f"   唯一值: {sample_df['地区'].unique()}")
    print(f"   唯一值数量: {sample_df['地区'].nunique()}")
    print(f"   值计数:\n{sample_df['地区'].value_counts()}")

    return (sample_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🎯 3. 数据选择和索引

    Pandas提供了多种方式来选择和访问数据。
    """
    )
    return


@app.cell
def _(sample_df):
    print("=" * 60)
    print("🎯 数据选择示例")
    print("=" * 60)

    # 1. 选择单列
    print("\n1️⃣ 选择单列:")
    print(sample_df['销售额'].head())

    # 2. 选择多列
    print("\n2️⃣ 选择多列:")
    print(sample_df[['日期', '销售额', '地区']].head())

    # 3. 使用loc按标签选择
    print("\n3️⃣ 使用loc选择前3行:")
    print(sample_df.loc[0:2, ['日期', '销售额']])

    # 4. 使用iloc按位置选择
    print("\n4️⃣ 使用iloc选择前3行的前2列:")
    print(sample_df.iloc[0:3, 0:2])

    # 5. 条件过滤
    print("\n5️⃣ 条件过滤（销售额>5000）:")
    high_sales = sample_df[sample_df['销售额'] > 5000]
    print(f"   符合条件的记录数: {len(high_sales)}")
    print(high_sales.head())

    # 6. 多条件过滤
    print("\n6️⃣ 多条件过滤（销售额>5000 且 地区='北区'）:")
    complex_filter = sample_df[(sample_df['销售额'] > 5000) & 
                                (sample_df['地区'] == '北区')]
    print(f"   符合条件的记录数: {len(complex_filter)}")
    print(complex_filter.head())

    # 7. 使用isin过滤
    print("\n7️⃣ 使用isin过滤（产品为A或B）:")
    product_filter = sample_df[sample_df['产品'].isin(['产品A', '产品B'])]
    print(f"   符合条件的记录数: {len(product_filter)}")

    return complex_filter, high_sales, product_filter


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧹 4. 数据清洗

    数据清洗是数据分析的重要步骤，包括处理缺失值、重复值和异常值。
    """
    )
    return


@app.cell
def _(np, pd):
    print("=" * 60)
    print("🧹 数据清洗示例")
    print("=" * 60)

    # 创建包含缺失值和重复值的数据
    dirty_data = pd.DataFrame({
        '姓名': ['张三', '李四', '王五', '张三', '赵六', None, '钱七'],
        '年龄': [25, 30, None, 25, 28, 32, 35],
        '城市': ['北京', '上海', '广州', '北京', None, '杭州', '深圳'],
        '薪资': [15000, 18000, 16000, 15000, 17000, 19000, None]
    })

    print("\n原始数据（包含缺失值和重复值）:")
    print(dirty_data)

    # 1. 检查缺失值
    print("\n1️⃣ 检查缺失值:")
    print(f"   每列缺失值数量:\n{dirty_data.isnull().sum()}")
    print(f"   总缺失值数量: {dirty_data.isnull().sum().sum()}")

    # 2. 删除包含缺失值的行
    print("\n2️⃣ 删除包含缺失值的行:")
    cleaned_dropna = dirty_data.dropna()
    print(cleaned_dropna)

    # 3. 填充缺失值
    print("\n3️⃣ 填充缺失值:")
    filled_data = dirty_data.copy()
    filled_data['年龄'].fillna(filled_data['年龄'].mean(), inplace=True)
    filled_data['城市'].fillna('未知', inplace=True)
    filled_data['薪资'].fillna(filled_data['薪资'].median(), inplace=True)
    filled_data['姓名'].fillna('匿名', inplace=True)
    print(filled_data)

    # 4. 检查重复值
    print("\n4️⃣ 检查重复值:")
    print(f"   重复行数: {dirty_data.duplicated().sum()}")
    print(f"   重复的行:")
    print(dirty_data[dirty_data.duplicated(keep=False)])

    # 5. 删除重复值
    print("\n5️⃣ 删除重复值:")
    deduped_data = dirty_data.drop_duplicates()
    print(deduped_data)

    # 6. 数据类型转换
    print("\n6️⃣ 数据类型转换:")
    type_converted = filled_data.copy()
    type_converted['年龄'] = type_converted['年龄'].astype(int)
    print(f"   转换后的数据类型:\n{type_converted.dtypes}")

    return cleaned_dropna, deduped_data, dirty_data, filled_data, type_converted


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🔄 5. 数据转换

    数据转换包括排序、映射、应用函数等操作。
    """
    )
    return


@app.cell
def _(np, pd, sample_df):
    print("=" * 60)
    print("🔄 数据转换示例")
    print("=" * 60)

    # 1. 排序
    print("\n1️⃣ 按销售额降序排序:")
    sorted_df = sample_df.sort_values('销售额', ascending=False)
    print(sorted_df.head())

    # 2. 多列排序
    print("\n2️⃣ 按地区升序、销售额降序排序:")
    multi_sorted = sample_df.sort_values(['地区', '销售额'],
                                         ascending=[True, False])
    print(multi_sorted.head(10))

    # 3. 添加新列
    print("\n3️⃣ 添加利润列:")
    transform_df = sample_df.copy()
    transform_df['利润'] = transform_df['销售额'] - transform_df['成本']
    transform_df['利润率'] = (transform_df['利润'] / transform_df['销售额'] * 100).round(2)
    print(transform_df[['日期', '销售额', '成本', '利润', '利润率']].head())

    # 4. 使用apply应用函数
    print("\n4️⃣ 使用apply应用函数:")
    def categorize_sales(sales):
        if sales >= 7000:
            return '高'
        elif sales >= 4000:
            return '中'
        else:
            return '低'

    transform_df['销售等级'] = transform_df['销售额'].apply(categorize_sales)
    print(transform_df[['销售额', '销售等级']].head(10))

    # 5. 使用map映射
    print("\n5️⃣ 使用map映射:")
    region_map = {'北区': 'North', '南区': 'South', '东区': 'East', '西区': 'West'}
    transform_df['Region_EN'] = transform_df['地区'].map(region_map)
    print(transform_df[['地区', 'Region_EN']].head())

    # 6. 使用replace替换值
    print("\n6️⃣ 使用replace替换值:")
    replaced_df = transform_df.copy()
    replaced_df['产品'] = replaced_df['产品'].replace({
        '产品A': 'Product-A',
        '产品B': 'Product-B',
        '产品C': 'Product-C'
    })
    print(replaced_df[['产品']].head())

    return categorize_sales, multi_sorted, region_map, replaced_df, sorted_df, transform_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📊 6. 数据聚合和分组

    分组聚合是数据分析中最常用的操作之一。
    """
    )
    return


@app.cell
def _(sample_df):
    print("=" * 60)
    print("📊 数据聚合和分组示例")
    print("=" * 60)

    # 1. 按单列分组聚合
    print("\n1️⃣ 按地区分组，计算平均销售额:")
    region_avg = sample_df.groupby('地区')['销售额'].mean().round(2)
    print(region_avg)

    # 2. 按多列分组
    print("\n2️⃣ 按地区和产品分组，计算总销售额:")
    multi_group = sample_df.groupby(['地区', '产品'])['销售额'].sum()
    print(multi_group)

    # 3. 多种聚合函数
    print("\n3️⃣ 按地区分组，应用多种聚合函数:")
    agg_result = sample_df.groupby('地区')['销售额'].agg(['sum', 'mean', 'min', 'max', 'count'])
    agg_result.columns = ['总销售额', '平均销售额', '最小销售额', '最大销售额', '记录数']
    print(agg_result)

    # 4. 对不同列应用不同聚合函数
    print("\n4️⃣ 对不同列应用不同聚合函数:")
    complex_agg = sample_df.groupby('地区').agg({
        '销售额': ['sum', 'mean'],
        '成本': ['sum', 'mean']
    }).round(2)
    print(complex_agg)

    # 5. 使用transform
    print("\n5️⃣ 使用transform添加组内平均值:")
    transform_result = sample_df.copy()
    transform_result['地区平均销售额'] = sample_df.groupby('地区')['销售额'].transform('mean').round(2)
    print(transform_result[['地区', '销售额', '地区平均销售额']].head(10))

    # 6. 透视表
    print("\n6️⃣ 创建透视表:")
    pivot_result = sample_df.pivot_table(
        values='销售额',
        index='地区',
        columns='产品',
        aggfunc='mean',
        fill_value=0
    ).round(2)
    print(pivot_result)

    return agg_result, complex_agg, multi_group, pivot_result, region_avg, transform_result


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🔗 7. 数据合并和连接

    Pandas提供了多种方式来合并和连接数据集。
    """
    )
    return


@app.cell
def _(pd):
    print("=" * 60)
    print("🔗 数据合并和连接示例")
    print("=" * 60)

    # 创建示例数据
    employees = pd.DataFrame({
        '员工ID': [1, 2, 3, 4],
        '姓名': ['张三', '李四', '王五', '赵六'],
        '部门ID': [101, 102, 101, 103]
    })

    departments = pd.DataFrame({
        '部门ID': [101, 102, 103, 104],
        '部门名称': ['技术部', '销售部', '人事部', '财务部']
    })

    salaries = pd.DataFrame({
        '员工ID': [1, 2, 3, 5],
        '薪资': [15000, 18000, 16000, 20000]
    })

    print("\n员工表:")
    print(employees)
    print("\n部门表:")
    print(departments)
    print("\n薪资表:")
    print(salaries)

    # 1. 内连接（inner join）
    print("\n1️⃣ 内连接（员工和部门）:")
    inner_join = pd.merge(employees, departments, on='部门ID', how='inner')
    print(inner_join)

    # 2. 左连接（left join）
    print("\n2️⃣ 左连接（员工和薪资）:")
    left_join = pd.merge(employees, salaries, on='员工ID', how='left')
    print(left_join)

    # 3. 右连接（right join）
    print("\n3️⃣ 右连接（员工和薪资）:")
    right_join = pd.merge(employees, salaries, on='员工ID', how='right')
    print(right_join)

    # 4. 外连接（outer join）
    print("\n4️⃣ 外连接（员工和薪资）:")
    outer_join = pd.merge(employees, salaries, on='员工ID', how='outer')
    print(outer_join)

    # 5. 多表连接
    print("\n5️⃣ 多表连接:")
    full_info = pd.merge(employees, departments, on='部门ID')
    full_info = pd.merge(full_info, salaries, on='员工ID', how='left')
    print(full_info)

    # 6. concat拼接
    print("\n6️⃣ 垂直拼接（concat）:")
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
    concat_result = pd.concat([df1, df2], ignore_index=True)
    print(concat_result)

    return concat_result, departments, df1, df2, employees, full_info, inner_join, left_join, outer_join, right_join, salaries


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🔀 8. 数据重塑

    数据重塑包括透视、堆叠、融合等操作。
    """
    )
    return


@app.cell
def _(pd):
    print("=" * 60)
    print("🔀 数据重塑示例")
    print("=" * 60)

    # 创建示例数据
    sales_data = pd.DataFrame({
        '日期': ['2024-01', '2024-01', '2024-02', '2024-02'],
        '产品': ['A', 'B', 'A', 'B'],
        '销售额': [100, 150, 120, 180]
    })

    print("\n原始数据:")
    print(sales_data)

    # 1. pivot - 透视
    print("\n1️⃣ 使用pivot透视:")
    pivoted = sales_data.pivot(index='日期', columns='产品', values='销售额')
    print(pivoted)

    # 2. melt - 融合（pivot的逆操作）
    print("\n2️⃣ 使用melt融合:")
    melted = pivoted.reset_index().melt(id_vars='日期',
                                         var_name='产品',
                                         value_name='销售额')
    print(melted)

    # 3. stack - 堆叠
    print("\n3️⃣ 使用stack堆叠:")
    stacked = pivoted.stack()
    print(stacked)

    # 4. unstack - 反堆叠
    print("\n4️⃣ 使用unstack反堆叠:")
    unstacked = stacked.unstack()
    print(unstacked)

    # 5. 宽格式转长格式
    print("\n5️⃣ 宽格式转长格式:")
    wide_data = pd.DataFrame({
        '学生': ['张三', '李四', '王五'],
        '语文': [85, 90, 88],
        '数学': [92, 88, 95],
        '英语': [78, 85, 90]
    })
    print("宽格式:")
    print(wide_data)

    long_data = wide_data.melt(id_vars='学生',
                                var_name='科目',
                                value_name='成绩')
    print("\n长格式:")
    print(long_data)

    return long_data, melted, pivoted, sales_data, stacked, unstacked, wide_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ⏰ 9. 时间序列处理

    Pandas提供了强大的时间序列处理功能。
    """
    )
    return


@app.cell
def _(np, pd):
    print("=" * 60)
    print("⏰ 时间序列处理示例")
    print("=" * 60)

    # 1. 创建日期范围
    print("\n1️⃣ 创建日期范围:")
    date_range = pd.date_range('2024-01-01', periods=10, freq='D')
    print(date_range)

    # 2. 创建时间序列数据
    print("\n2️⃣ 创建时间序列数据:")
    ts_data = pd.DataFrame({
        '日期': pd.date_range('2024-01-01', periods=30, freq='D'),
        '销售额': np.random.randint(1000, 5000, 30),
        '访客数': np.random.randint(100, 500, 30)
    })
    ts_data.set_index('日期', inplace=True)
    print(ts_data.head(10))

    # 3. 日期时间属性提取
    print("\n3️⃣ 提取日期时间属性:")
    ts_extract = ts_data.copy()
    ts_extract['年'] = ts_extract.index.year
    ts_extract['月'] = ts_extract.index.month
    ts_extract['日'] = ts_extract.index.day
    ts_extract['星期'] = ts_extract.index.dayofweek
    ts_extract['星期名'] = ts_extract.index.day_name()
    print(ts_extract.head())

    # 4. 时间序列重采样
    print("\n4️⃣ 按周重采样:")
    weekly_data = ts_data.resample('W').sum()
    print(weekly_data)

    # 5. 滚动窗口计算
    print("\n5️⃣ 计算7天移动平均:")
    ts_rolling = ts_data.copy()
    ts_rolling['销售额_7日均值'] = ts_rolling['销售额'].rolling(window=7).mean().round(2)
    print(ts_rolling.head(10))

    # 6. 时间偏移
    print("\n6️⃣ 时间偏移:")
    ts_shift = ts_data.copy()
    ts_shift['昨日销售额'] = ts_shift['销售额'].shift(1)
    ts_shift['销售额变化'] = ts_shift['销售额'] - ts_shift['昨日销售额']
    print(ts_shift.head(10))

    return date_range, ts_data, ts_extract, ts_rolling, ts_shift, weekly_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 💾 10. 数据IO操作

    Pandas支持读写多种数据格式。
    """
    )
    return


@app.cell
def _(pd, sample_df):
    print("=" * 60)
    print("💾 数据IO操作示例")
    print("=" * 60)

    # 1. CSV操作
    print("\n1️⃣ CSV操作:")
    csv_file = 'temp_data.csv'
    sample_df.head(10).to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"   已保存到: {csv_file}")

    read_csv = pd.read_csv(csv_file)
    print(f"   读取的数据形状: {read_csv.shape}")
    print(read_csv.head(3))

    # 2. Excel操作（需要openpyxl库）
    print("\n2️⃣ Excel操作:")
    try:
        excel_file = 'temp_data.xlsx'
        sample_df.head(10).to_excel(excel_file, index=False, sheet_name='销售数据')
        print(f"   已保存到: {excel_file}")

        read_excel = pd.read_excel(excel_file, sheet_name='销售数据')
        print(f"   读取的数据形状: {read_excel.shape}")
    except ImportError:
        print("   需要安装openpyxl: pip install openpyxl")

    # 3. JSON操作
    print("\n3️⃣ JSON操作:")
    json_file = 'temp_data.json'
    sample_df.head(5).to_json(json_file, orient='records', force_ascii=False, indent=2)
    print(f"   已保存到: {json_file}")

    read_json = pd.read_json(json_file)
    print(f"   读取的数据形状: {read_json.shape}")

    # 4. 从字典列表创建
    print("\n4️⃣ 从字典列表创建DataFrame:")
    dict_list = [
        {'姓名': '张三', '年龄': 25, '城市': '北京'},
        {'姓名': '李四', '年龄': 30, '城市': '上海'},
        {'姓名': '王五', '年龄': 35, '城市': '广州'}
    ]
    df_from_dict_list = pd.DataFrame(dict_list)
    print(df_from_dict_list)

    # 5. 转换为字典
    print("\n5️⃣ DataFrame转换为字典:")
    to_dict_records = df_from_dict_list.to_dict('records')
    print(f"   records格式: {to_dict_records}")

    to_dict_list = df_from_dict_list.to_dict('list')
    print(f"   list格式: {to_dict_list}")

    return csv_file, df_from_dict_list, dict_list, excel_file, json_file, read_csv, read_excel, read_json, to_dict_list, to_dict_records


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🎯 11. 常用技巧和最佳实践

    一些实用的Pandas技巧和最佳实践。
    """
    )
    return


@app.cell
def _(np, pd):
    print("=" * 60)
    print("🎯 常用技巧示例")
    print("=" * 60)

    # 1. 链式操作
    print("\n1️⃣ 链式操作:")
    chained_result = (pd.DataFrame({
        '姓名': ['张三', '李四', '王五', '赵六'],
        '年龄': [25, 30, 35, 28],
        '薪资': [15000, 18000, 16000, 17000]
    })
    .assign(税后薪资=lambda x: x['薪资'] * 0.8)
    .query('年龄 > 26')
    .sort_values('薪资', ascending=False)
    .reset_index(drop=True)
    )
    print(chained_result)

    # 2. 使用query进行过滤
    print("\n2️⃣ 使用query进行过滤:")
    query_df = pd.DataFrame({
        '产品': ['A', 'B', 'C', 'A', 'B'],
        '销售额': [100, 200, 150, 180, 220],
        '地区': ['北', '南', '东', '西', '北']
    })
    filtered = query_df.query('销售额 > 150 and 地区 == "北"')
    print(filtered)

    # 3. 使用eval进行计算
    print("\n3️⃣ 使用eval进行计算:")
    eval_df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })
    eval_df.eval('D = A + B * C', inplace=True)
    print(eval_df)

    # 4. 内存优化
    print("\n4️⃣ 内存优化:")
    memory_df = pd.DataFrame({
        '整数列': np.random.randint(0, 100, 1000),
        '分类列': np.random.choice(['A', 'B', 'C'], 1000)
    })

    print(f"   优化前内存: {memory_df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    memory_df['整数列'] = memory_df['整数列'].astype('int8')
    memory_df['分类列'] = memory_df['分类列'].astype('category')

    print(f"   优化后内存: {memory_df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    # 5. 使用cut进行分箱
    print("\n5️⃣ 使用cut进行分箱:")
    cut_df = pd.DataFrame({
        '年龄': [18, 25, 35, 45, 55, 65, 75]
    })
    cut_df['年龄段'] = pd.cut(cut_df['年龄'],
                              bins=[0, 30, 50, 100],
                              labels=['青年', '中年', '老年'])
    print(cut_df)

    # 6. 使用qcut进行等频分箱
    print("\n6️⃣ 使用qcut进行等频分箱:")
    qcut_df = pd.DataFrame({
        '分数': [60, 70, 75, 80, 85, 90, 95, 100]
    })
    qcut_df['等级'] = pd.qcut(qcut_df['分数'],
                              q=4,
                              labels=['D', 'C', 'B', 'A'])
    print(qcut_df)

    return chained_result, cut_df, eval_df, filtered, memory_df, qcut_df, query_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🚀 12. 完整实战案例

    让我们通过一个完整的数据分析案例来综合运用Pandas的各种功能。

    ### 场景：电商销售数据分析

    我们将分析一个电商平台的销售数据，包括：
    - 数据加载和清洗
    - 探索性数据分析
    - 数据转换和特征工程
    - 数据聚合和可视化
    """
    )
    return


@app.cell
def _(np, pd):
    print("=" * 60)
    print("🚀 完整实战案例：电商销售数据分析")
    print("=" * 60)

    # 1. 创建模拟数据
    print("\n📊 步骤1：创建模拟电商销售数据")
    np.random.seed(42)
    n_records = 1000

    ecommerce_data = pd.DataFrame({
        '订单ID': range(1, n_records + 1),
        '日期': pd.date_range('2024-01-01', periods=n_records, freq='H'),
        '用户ID': np.random.randint(1000, 2000, n_records),
        '产品类别': np.random.choice(['电子产品', '服装', '食品', '图书', '家居'], n_records),
        '产品名称': [f'产品{i}' for i in np.random.randint(1, 100, n_records)],
        '数量': np.random.randint(1, 10, n_records),
        '单价': np.random.uniform(10, 1000, n_records).round(2),
        '地区': np.random.choice(['华北', '华东', '华南', '华中', '西南', '东北'], n_records),
        '支付方式': np.random.choice(['支付宝', '微信', '信用卡', '货到付款'], n_records)
    })

    # 添加一些缺失值
    ecommerce_data.loc[np.random.choice(ecommerce_data.index, 20), '地区'] = None
    ecommerce_data.loc[np.random.choice(ecommerce_data.index, 15), '支付方式'] = None

    print(f"数据形状: {ecommerce_data.shape}")
    print("\n前5条记录:")
    print(ecommerce_data.head())

    # 2. 数据清洗
    print("\n🧹 步骤2：数据清洗")

    # 检查缺失值
    print(f"缺失值统计:\n{ecommerce_data.isnull().sum()}")

    # 填充缺失值
    ecommerce_data['地区'].fillna('未知', inplace=True)
    ecommerce_data['支付方式'].fillna('其他', inplace=True)

    # 检查重复值
    duplicates = ecommerce_data.duplicated().sum()
    print(f"重复记录数: {duplicates}")

    print("✅ 数据清洗完成")

    # 3. 特征工程
    print("\n🔧 步骤3：特征工程")

    # 计算总金额
    ecommerce_data['总金额'] = (ecommerce_data['数量'] * ecommerce_data['单价']).round(2)

    # 提取时间特征
    ecommerce_data['年'] = ecommerce_data['日期'].dt.year
    ecommerce_data['月'] = ecommerce_data['日期'].dt.month
    ecommerce_data['日'] = ecommerce_data['日期'].dt.day
    ecommerce_data['小时'] = ecommerce_data['日期'].dt.hour
    ecommerce_data['星期'] = ecommerce_data['日期'].dt.dayofweek
    ecommerce_data['是否周末'] = ecommerce_data['星期'].isin([5, 6])

    print("新增特征:")
    print(ecommerce_data[['日期', '总金额', '年', '月', '日', '小时', '星期', '是否周末']].head())

    return duplicates, ecommerce_data, n_records


@app.cell
def _(ecommerce_data):
    print("=" * 60)
    print("📈 步骤4：探索性数据分析")
    print("=" * 60)

    # 1. 基本统计信息
    print("\n1️⃣ 数值列的描述性统计:")
    print(ecommerce_data[['数量', '单价', '总金额']].describe())

    # 2. 按产品类别分析
    print("\n2️⃣ 按产品类别分析:")
    category_analysis = ecommerce_data.groupby('产品类别').agg({
        '订单ID': 'count',
        '总金额': ['sum', 'mean', 'max'],
        '数量': 'sum'
    }).round(2)
    category_analysis.columns = ['订单数', '总销售额', '平均订单金额', '最大订单金额', '总销量']
    category_analysis = category_analysis.sort_values('总销售额', ascending=False)
    print(category_analysis)

    # 3. 按地区分析
    print("\n3️⃣ 按地区分析:")
    region_analysis = ecommerce_data.groupby('地区').agg({
        '订单ID': 'count',
        '总金额': 'sum'
    }).round(2)
    region_analysis.columns = ['订单数', '总销售额']
    region_analysis = region_analysis.sort_values('总销售额', ascending=False)
    print(region_analysis)

    # 4. 按支付方式分析
    print("\n4️⃣ 按支付方式分析:")
    payment_analysis = ecommerce_data['支付方式'].value_counts()
    payment_pct = (payment_analysis / len(ecommerce_data) * 100).round(2)
    payment_df = pd.DataFrame({
        '订单数': payment_analysis,
        '占比(%)': payment_pct
    })
    print(payment_df)

    # 5. 时间趋势分析
    print("\n5️⃣ 按日期分析销售趋势:")
    daily_sales = ecommerce_data.groupby(ecommerce_data['日期'].dt.date).agg({
        '订单ID': 'count',
        '总金额': 'sum'
    }).round(2)
    daily_sales.columns = ['订单数', '总销售额']
    print(daily_sales.head(10))

    # 6. 按小时分析
    print("\n6️⃣ 按小时分析订单分布:")
    hourly_orders = ecommerce_data.groupby('小时')['订单ID'].count()
    print(hourly_orders)

    # 7. 周末vs工作日
    print("\n7️⃣ 周末vs工作日对比:")
    weekend_analysis = ecommerce_data.groupby('是否周末').agg({
        '订单ID': 'count',
        '总金额': ['sum', 'mean']
    }).round(2)
    weekend_analysis.columns = ['订单数', '总销售额', '平均订单金额']
    weekend_analysis.index = ['工作日', '周末']
    print(weekend_analysis)

    return category_analysis, daily_sales, hourly_orders, payment_analysis, payment_df, payment_pct, region_analysis, weekend_analysis


@app.cell
def _(ecommerce_data):
    print("=" * 60)
    print("🎯 步骤5：高级分析")
    print("=" * 60)

    # 1. 用户行为分析
    print("\n1️⃣ 用户购买行为分析:")
    user_behavior = ecommerce_data.groupby('用户ID').agg({
        '订单ID': 'count',
        '总金额': 'sum',
        '产品类别': lambda x: x.nunique()
    }).round(2)
    user_behavior.columns = ['购买次数', '总消费金额', '购买类别数']
    user_behavior['平均订单金额'] = (user_behavior['总消费金额'] / user_behavior['购买次数']).round(2)

    print("用户行为统计:")
    print(user_behavior.describe())

    print("\n消费金额TOP10用户:")
    print(user_behavior.nlargest(10, '总消费金额'))

    # 2. RFM分析（简化版）
    print("\n2️⃣ RFM分析:")
    latest_date = ecommerce_data['日期'].max()

    rfm = ecommerce_data.groupby('用户ID').agg({
        '日期': lambda x: (latest_date - x.max()).days,  # Recency
        '订单ID': 'count',  # Frequency
        '总金额': 'sum'  # Monetary
    }).round(2)
    rfm.columns = ['最近购买天数', '购买频次', '总消费金额']

    # RFM评分（简化）- 使用cut代替qcut以避免分箱错误
    try:
        rfm['R评分'] = pd.qcut(rfm['最近购买天数'], 5, labels=[5,4,3,2,1], duplicates='drop')
    except ValueError:
        rfm['R评分'] = pd.cut(rfm['最近购买天数'], 5, labels=[5,4,3,2,1])

    try:
        rfm['F评分'] = pd.qcut(rfm['购买频次'], 5, labels=[1,2,3,4,5], duplicates='drop')
    except ValueError:
        rfm['F评分'] = pd.cut(rfm['购买频次'], 5, labels=[1,2,3,4,5])

    try:
        rfm['M评分'] = pd.qcut(rfm['总消费金额'], 5, labels=[1,2,3,4,5], duplicates='drop')
    except ValueError:
        rfm['M评分'] = pd.cut(rfm['总消费金额'], 5, labels=[1,2,3,4,5])

    print(rfm.head(10))

    # 3. 产品关联分析
    print("\n3️⃣ 产品类别组合分析:")
    user_categories = ecommerce_data.groupby('用户ID')['产品类别'].apply(list)

    # 统计类别组合
    from itertools import combinations
    category_pairs = []
    for categories in user_categories:
        if len(categories) >= 2:
            for pair in combinations(set(categories), 2):
                category_pairs.append(tuple(sorted(pair)))

    if category_pairs:
        pair_counts = pd.Series(category_pairs).value_counts().head(10)
        print("常见产品类别组合:")
        print(pair_counts)

    # 4. 客单价分析
    print("\n4️⃣ 客单价分布分析:")
    avg_order_value = ecommerce_data.groupby('订单ID')['总金额'].sum()

    print(f"平均客单价: {avg_order_value.mean():.2f}")
    print(f"中位数客单价: {avg_order_value.median():.2f}")
    print(f"最高客单价: {avg_order_value.max():.2f}")
    print(f"最低客单价: {avg_order_value.min():.2f}")

    return avg_order_value, category_pairs, combinations, latest_date, pair_counts, rfm, user_behavior, user_categories


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📚 13. Pandas最佳实践总结

    ### 性能优化技巧

    1. **使用向量化操作**：避免使用循环，使用Pandas内置的向量化方法
    2. **选择合适的数据类型**：使用`category`类型存储重复字符串，使用`int8`等节省内存
    3. **使用`query()`和`eval()`**：对于大数据集，这些方法更高效
    4. **分块读取大文件**：使用`chunksize`参数分块读取
    5. **使用`inplace=True`谨慎**：虽然节省内存，但可能导致意外结果

    ### 代码可读性

    1. **使用链式操作**：让代码更简洁易读
    2. **合理命名变量**：使用描述性的变量名
    3. **添加注释**：解释复杂的操作逻辑
    4. **拆分复杂操作**：将复杂的数据处理拆分成多个步骤

    ### 数据质量

    1. **始终检查数据**：使用`info()`, `describe()`, `head()`等方法
    2. **处理缺失值**：明确缺失值的处理策略
    3. **验证数据类型**：确保每列的数据类型正确
    4. **检查重复值**：及时发现和处理重复数据
    5. **数据验证**：使用断言验证数据的合理性

    ### 常用方法速查表

    #### 📊 数据创建

    | 方法 | 说明 | 示例 |
    |------|------|------|
    | `pd.DataFrame()` | 创建DataFrame | `pd.DataFrame({'A': [1, 2], 'B': [3, 4]})` |
    | `pd.Series()` | 创建Series | `pd.Series([1, 2, 3, 4])` |
    | `pd.read_csv()` | 读取CSV文件 | `pd.read_csv('data.csv')` |
    | `pd.read_excel()` | 读取Excel文件 | `pd.read_excel('data.xlsx')` |
    | `pd.read_json()` | 读取JSON文件 | `pd.read_json('data.json')` |
    | `pd.read_sql()` | 从SQL数据库读取 | `pd.read_sql(query, connection)` |

    #### 👀 数据查看

    | 方法 | 说明 | 示例 |
    |------|------|------|
    | `df.head(n)` | 查看前n行（默认5行） | `df.head(10)` |
    | `df.tail(n)` | 查看后n行（默认5行） | `df.tail(10)` |
    | `df.sample(n)` | 随机抽样n行 | `df.sample(5)` |
    | `df.info()` | 查看数据信息和类型 | `df.info()` |
    | `df.describe()` | 描述性统计 | `df.describe()` |
    | `df.shape` | 数据形状（行数，列数） | `df.shape` |
    | `df.columns` | 列名列表 | `df.columns` |
    | `df.dtypes` | 各列数据类型 | `df.dtypes` |
    | `df.index` | 索引信息 | `df.index` |

    #### 🎯 数据选择

    | 方法 | 说明 | 示例 |
    |------|------|------|
    | `df['col']` | 选择单列 | `df['姓名']` |
    | `df[['col1', 'col2']]` | 选择多列 | `df[['姓名', '年龄']]` |
    | `df.loc[]` | 按标签选择行列 | `df.loc[0:5, ['姓名', '年龄']]` |
    | `df.iloc[]` | 按位置选择行列 | `df.iloc[0:5, 0:2]` |
    | `df[condition]` | 条件过滤 | `df[df['年龄'] > 25]` |
    | `df.query()` | 查询表达式过滤 | `df.query('年龄 > 25 and 城市 == "北京"')` |
    | `df.isin()` | 成员检查 | `df[df['城市'].isin(['北京', '上海'])]` |

    #### 🧹 数据清洗

    | 方法 | 说明 | 示例 |
    |------|------|------|
    | `df.isnull()` | 检查缺失值 | `df.isnull().sum()` |
    | `df.notnull()` | 检查非缺失值 | `df.notnull()` |
    | `df.dropna()` | 删除缺失值 | `df.dropna(axis=0)` |
    | `df.fillna()` | 填充缺失值 | `df.fillna(0)` |
    | `df.duplicated()` | 检查重复值 | `df.duplicated()` |
    | `df.drop_duplicates()` | 删除重复值 | `df.drop_duplicates()` |
    | `df.replace()` | 替换值 | `df.replace({'A': 'B'})` |
    | `df.astype()` | 转换数据类型 | `df['年龄'].astype(int)` |

    #### 🔄 数据转换

    | 方法 | 说明 | 示例 |
    |------|------|------|
    | `df.sort_values()` | 按值排序 | `df.sort_values('年龄', ascending=False)` |
    | `df.sort_index()` | 按索引排序 | `df.sort_index()` |
    | `df.apply()` | 应用函数 | `df['年龄'].apply(lambda x: x + 1)` |
    | `df.map()` | 映射值 | `df['性别'].map({'M': '男', 'F': '女'})` |
    | `df.assign()` | 添加新列 | `df.assign(新列=df['A'] + df['B'])` |
    | `df.rename()` | 重命名列 | `df.rename(columns={'old': 'new'})` |
    | `df.drop()` | 删除行或列 | `df.drop(['列名'], axis=1)` |
    | `df.reset_index()` | 重置索引 | `df.reset_index(drop=True)` |

    #### 📊 数据聚合

    | 方法 | 说明 | 示例 |
    |------|------|------|
    | `df.groupby()` | 分组 | `df.groupby('类别')['销售额'].sum()` |
    | `df.agg()` | 聚合函数 | `df.agg({'A': 'sum', 'B': 'mean'})` |
    | `df.pivot_table()` | 透视表 | `df.pivot_table(values='销售额', index='地区', columns='产品')` |
    | `df.value_counts()` | 值计数 | `df['类别'].value_counts()` |
    | `df.sum()` | 求和 | `df.sum()` |
    | `df.mean()` | 平均值 | `df.mean()` |
    | `df.median()` | 中位数 | `df.median()` |
    | `df.std()` | 标准差 | `df.std()` |
    | `df.min()` / `df.max()` | 最小值/最大值 | `df.min()` |
    | `df.count()` | 计数 | `df.count()` |

    #### 🔗 数据合并

    | 方法 | 说明 | 示例 |
    |------|------|------|
    | `pd.merge()` | 合并（类似SQL JOIN） | `pd.merge(df1, df2, on='key', how='inner')` |
    | `pd.concat()` | 拼接 | `pd.concat([df1, df2], axis=0)` |
    | `df.join()` | 连接 | `df1.join(df2, on='key')` |
    | `df.append()` | 追加行 | `df.append(new_row, ignore_index=True)` |

    #### 🔀 数据重塑

    | 方法 | 说明 | 示例 |
    |------|------|------|
    | `df.pivot()` | 透视（宽格式） | `df.pivot(index='日期', columns='产品', values='销售额')` |
    | `df.melt()` | 融合（长格式） | `df.melt(id_vars='日期', value_vars=['A', 'B'])` |
    | `df.stack()` | 堆叠（列转行） | `df.stack()` |
    | `df.unstack()` | 反堆叠（行转列） | `df.unstack()` |
    | `df.transpose()` / `df.T` | 转置 | `df.T` |

    #### ⏰ 时间序列

    | 方法 | 说明 | 示例 |
    |------|------|------|
    | `pd.date_range()` | 创建日期范围 | `pd.date_range('2024-01-01', periods=10, freq='D')` |
    | `pd.to_datetime()` | 转换为日期时间 | `pd.to_datetime(df['日期'])` |
    | `df.resample()` | 重采样 | `df.resample('W').sum()` |
    | `df.rolling()` | 滚动窗口 | `df.rolling(window=7).mean()` |
    | `df.shift()` | 时间偏移 | `df.shift(1)` |
    | `df.diff()` | 差分 | `df.diff()` |

    #### 💾 数据IO

    | 方法 | 说明 | 示例 |
    |------|------|------|
    | `df.to_csv()` | 保存为CSV | `df.to_csv('data.csv', index=False)` |
    | `df.to_excel()` | 保存为Excel | `df.to_excel('data.xlsx', index=False)` |
    | `df.to_json()` | 保存为JSON | `df.to_json('data.json')` |
    | `df.to_sql()` | 保存到SQL数据库 | `df.to_sql('table_name', connection)` |
    | `df.to_dict()` | 转换为字典 | `df.to_dict('records')` |

    ### 学习资源

    - 📖 [Pandas官方文档](https://pandas.pydata.org/docs/)
    - 📚 [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
    - 💡 [Pandas Cookbook](https://pandas.pydata.org/docs/user_guide/cookbook.html)
    - 🎓 [Real Python Pandas Tutorials](https://realpython.com/learning-paths/pandas-data-science/)

    ### 总结

    Pandas是Python数据分析的核心工具，掌握它对于数据科学家和分析师至关重要。

    **关键要点**：
    - 🎯 理解Series和DataFrame的核心概念
    - 🔧 熟练使用数据选择、过滤、转换方法
    - 🧹 重视数据清洗和质量检查
    - 📊 掌握分组聚合和数据重塑
    - ⏰ 了解时间序列处理
    - 💾 熟悉各种数据IO操作
    - 🚀 注重性能优化和代码可读性

    通过本指南的学习和实践，你应该能够：
    - ✅ 使用Pandas进行数据加载和保存
    - ✅ 进行数据清洗和预处理
    - ✅ 执行复杂的数据转换和聚合
    - ✅ 进行探索性数据分析
    - ✅ 为机器学习准备数据

    继续实践，不断提升你的Pandas技能！🐼
    """
    )
    return


if __name__ == "__main__":
    app.run()

