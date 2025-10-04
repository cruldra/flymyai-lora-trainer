import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # MLOps完整蓝图：数据和管道工程（含实现）

    MLOps和LLMOps速成课程——第7部分

    ## 📚 本章概览

    在本章中，我们将深入探讨MLOps中的两个关键组件：

    1. **Apache Spark**：用于大规模分布式数据处理
    2. **Prefect**：用于工作流编排和调度

    这两个工具在现代MLOps流程中扮演着至关重要的角色，帮助我们处理海量数据并自动化复杂的机器学习管道。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🌟 Apache Spark分布式数据处理

    ### 为什么需要Spark？

    随着数据量的增长，单机工具（如Pandas、NumPy）可能会开始力不从心。这时就需要Apache Spark——一个广泛用于大数据处理的分布式计算引擎。

    ### Spark是什么？

    Spark是一个集群计算框架，提供了分布式数据结构（如弹性分布式数据集RDD和更高级的DataFrame）及其操作的API。

    **核心特点：**

    - **分布式计算**：将数据分布在多台机器上并行处理
    - **内存计算**：尽可能在内存中处理数据，速度快
    - **多语言支持**：Scala、Python（PySpark）、Java、R等
    - **丰富的生态**：包含Spark SQL、MLlib、GraphX等组件

    ### Spark在ML中的两个关键方面

    1. **DataFrame API**：类似于Pandas DataFrame，但是分布式的
    2. **Spark MLlib**：包含可以分布式运行的机器学习算法和Pipeline
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📊 Spark DataFrame

    Spark的DataFrame在概念上类似于分布在集群中的表。你可以执行类似SQL的操作：过滤、连接、分组等，Spark会自动并行化这些操作。

    ### 底层原理

    - 基于RDD构建
    - 通过Catalyst查询优化器提供优化
    - 数据被分区并分布在不同的worker节点上
    - 每个worker只处理自己的数据分区

    ### Spark在ML ETL中的应用

    许多数据工程管道使用Spark来完成繁重的工作：

    - 从数据湖读取数据
    - 连接大型表
    - 计算特征（如聚合）
    - 输出结果到存储（如Parquet文件）
    - 直接通过MLlib训练模型
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🔧 Spark ML Pipeline示例

    让我们通过一个完整的示例来理解Spark ML Pipeline的工作原理。

    ### 示例场景：时间序列回归

    我们将构建一个包含以下步骤的Pipeline：

    1. **数据生成**：创建合成的时间序列数据
    2. **时间分割**：按时间划分训练集和测试集
    3. **特征工程**：提取时间特征（如小时）
    4. **数据预处理**：缺失值填充
    5. **特征组装**：将特征组合成向量
    6. **模型训练**：线性回归
    7. **模型评估**：计算R²分数
    """
    )
    return


@app.cell
def _():
    # 注意：这是演示代码，需要安装PySpark
    # pip install pyspark

    print("🔧 Spark ML Pipeline演示")
    print("=" * 50)

    # 由于在本地环境中运行完整的Spark可能比较复杂，
    # 这里我们展示关键概念和代码结构

    spark_pipeline_code = """
    from pyspark.sql import SparkSession
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import Imputer, VectorAssembler
    from pyspark.ml.regression import LinearRegression
    from pyspark.ml.evaluation import RegressionEvaluator
    import pyspark.sql.functions as F

    # 1. 创建Spark会话
    spark = SparkSession.builder \\
        .appName("SimpleSparkMLPipeline") \\
        .master("local[*]") \\
        .config("spark.driver.memory", "4g") \\
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # 2. 生成合成数据
    n_rows = 10_000
    start_ts = int(F.unix_timestamp(F.lit("2024-01-01 00:00:00")).cast("long"))

    df = spark.range(n_rows) \\
        .withColumn("ts", start_ts + (F.col("id") * 60)) \\
        .withColumn("ds", F.from_unixtime(F.col("ts")).cast("timestamp")) \\
        .withColumn("feature_a", F.randn(seed=42)) \\
        .withColumn("feature_b", F.rand(seed=1337) * 10.0) \\
        .withColumn("y", 2.0*F.col("feature_a") + 0.3*F.col("feature_b") + F.randn(seed=7)*0.5) \\
        .drop("ts")

    # 3. 时间分割
    split_date = "2024-01-08 00:00:00"
    train = df.filter(F.col("ds") < split_date)
    test = df.filter(F.col("ds") >= split_date)

    # 4. 特征工程
    train = train.withColumn("hour", F.hour(F.col("ds")).cast("double"))
    test = test.withColumn("hour", F.hour(F.col("ds")).cast("double"))

    # 5. 定义Pipeline
    imputer = Imputer(
        inputCols=["hour", "feature_a", "feature_b"],
        outputCols=["hour_imp", "feature_a_imp", "feature_b_imp"]
    )

    assembler = VectorAssembler(
        inputCols=["hour_imp", "feature_a_imp", "feature_b_imp"],
        outputCol="features"
    )

    lr = LinearRegression(labelCol="y", featuresCol="features")

    pipeline = Pipeline(stages=[imputer, assembler, lr])

    # 6. 训练模型
    model = pipeline.fit(train)

    # 7. 评估模型
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(labelCol="y", predictionCol="prediction", metricName="r2")
    r2 = evaluator.evaluate(predictions)

    print(f"测试集R²分数: {r2:.4f}")

    spark.stop()
    """

    print("\n📝 Spark ML Pipeline代码结构：")
    print(spark_pipeline_code)

    return (spark_pipeline_code,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🔍 Spark Pipeline工作流程详解

    ### 步骤1：Spark会话初始化

    ```python
    spark = SparkSession.builder \\
        .appName("SimpleSparkMLPipeline") \\
        .master("local[*]") \\
        .config("spark.driver.memory", "4g") \\
        .getOrCreate()
    ```

    - 创建本地Spark会话
    - 使用所有可用CPU核心
    - 限制驱动内存为4GB
    - 设置日志级别为WARN

    ### 步骤2：数据生成

    - 创建10,000行时间序列数据
    - 每行间隔1分钟
    - 生成两个特征：feature_a（高斯分布）、feature_b（均匀分布）
    - 目标变量y = 2.0*feature_a + 0.3*feature_b + 噪声

    ### 步骤3：时间分割

    - 按时间戳分割训练集和测试集
    - 避免数据泄漏（未来信息不会影响过去的预测）

    ### 步骤4：特征工程

    - 从时间戳中提取小时特征
    - 这是一个确定性转换，不会导致泄漏

    ### 步骤5：Pipeline定义

    - **Imputer**：处理缺失值（虽然合成数据没有缺失值）
    - **VectorAssembler**：将特征组合成向量
    - **LinearRegression**：线性回归模型

    ### 步骤6：训练

    - Pipeline在训练集上fit
    - 所有转换器的参数都从训练集学习

    ### 步骤7：评估

    - 在测试集上transform
    - 计算R²分数
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ⚖️ Pandas vs Spark：何时使用哪个？

    ### Pandas的优势

    **适用场景：**
    - 数据量 < 几百万行
    - 数据大小 < 几GB
    - 单机内存足够
    - 快速原型开发
    - 探索性数据分析

    **特点：**
    - ✅ 低开销，启动快
    - ✅ 简单易用
    - ✅ 丰富的API
    - ✅ 即时执行（eager execution）
    - ❌ 受限于单机内存
    - ❌ 无法处理超大数据集

    ### Spark的优势

    **适用场景：**
    - 数据量 > 数十亿行
    - 数据分布在集群中
    - 需要复杂的shuffle操作
    - 数据存储在HDFS/数据湖
    - 需要分布式计算

    **特点：**
    - ✅ 可扩展到集群
    - ✅ 处理超大数据集
    - ✅ 容错机制
    - ✅ 懒执行（lazy execution）
    - ✅ 可以spill到磁盘
    - ❌ 启动开销大
    - ❌ 小数据集上可能更慢
    - ❌ 调试相对困难
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧪 Pandas vs Spark：懒执行演示

    让我们通过一个实际例子来理解Pandas的即时执行和Spark的懒执行的区别。

    ### 场景：生成大量数据

    我们将尝试生成1亿行数据，看看Pandas和Spark的表现。
    """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd

    print("🧪 Pandas vs Spark执行模式对比")
    print("=" * 50)

    # Pandas示例（小规模演示）
    print("\n1️⃣ Pandas（即时执行）:")
    print("   - 所有操作立即执行")
    print("   - 数据全部加载到内存")
    print("   - 大数据集会导致内存溢出")

    # 小规模演示
    N_small = 1_000_000
    print(f"\n   演示：生成{N_small:,}行数据")

    df_pandas = pd.DataFrame({
        'id': range(N_small),
        'value': np.random.randn(N_small)
    })

    print(f"   ✅ 成功！内存使用: {df_pandas.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Spark示例（概念说明）
    print("\n2️⃣ Spark（懒执行）:")
    print("   - 操作构建执行计划")
    print("   - 只有遇到Action才真正执行")
    print("   - 数据分区处理，可以spill到磁盘")
    print("   - 只返回最终结果，不是全部数据")

    spark_concept = """
    # Spark代码示例
    df_spark = spark.range(100_000_000)  # 这不会立即执行
    df_spark = df_spark.withColumn("value", F.randn())  # 仍然不执行
    count = df_spark.count()  # 这是Action，触发执行
    # 但只返回count值，不是全部数据
    """

    print(f"\n   代码示例：")
    print(spark_concept)

    print("\n💡 关键区别：")
    print("   - Pandas: 必须将全部数据加载到内存")
    print("   - Spark: 分区处理，只返回聚合结果")
    print("   - 在N=100,000,000时，Pandas会崩溃，Spark能完成")

    return N_small, df_pandas, np, pd, spark_concept


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📊 何时使用Spark？

    ### 使用Spark的场景

    1. **数据量巨大**
       - 数十亿条记录
       - 数据大小超过单机内存
       - 需要处理TB级别的数据

    2. **数据分布式存储**
       - 数据存储在HDFS
       - 数据在数据湖中
       - 需要从多个数据源读取

    3. **复杂的数据处理**
       - 大规模join操作
       - 复杂的聚合计算
       - 需要并行处理

    ### 实际案例

    **场景**：网站用户交互日志分析

    - 数据量：10亿条事件记录
    - 存储：Parquet格式在HDFS上
    - 任务：为流失模型创建用户特征（平均会话时间等）

    **为什么用Spark？**
    - Pandas无法处理如此大的数据量
    - Spark可以分布式group by用户并计算聚合
    - 更快、更高效

    ### 不需要Spark的场景

    - 数据 < 几百万行
    - 数据 < 几GB
    - 单机内存足够
    - 快速原型开发

    **注意**：前面的10,000行示例更适合用Pandas，代码仅用于演示Spark的使用方式。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ⚠️ Spark的局限性

    虽然Spark很强大，但它并非万能：

    ### 性能瓶颈

    - **网络和I/O**：可能成为瓶颈
    - **算法限制**：不是所有算法都能轻松并行化
    - **小数据开销**：对于小数据集，启动开销可能超过收益

    ### 调试困难

    - 分布式环境下调试更复杂
    - 需要熟悉日志系统
    - 性能问题诊断需要经验

    ### 学习曲线

    - 需要理解分布式计算概念
    - 需要了解Spark的执行模型
    - 配置和调优需要经验

    ### 总结

    Apache Spark将管道扩展到大数据规模，让团队能够实现分布式ETL甚至建模。在MLOps环境中，熟悉Spark意味着你可以创建利用分布式特性的管道，这在生产环境中处理海量数据时是必需的。

    **关键原则**：在需要时使用它。对于许多MLOps任务，小数据工具就足够了，但当你遇到大数据领域或需要并行处理的能力时，Spark（或类似框架）就变得不可或缺。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---

    ## 🔄 工作流编排与管理

    构建管道是一回事，可靠地按计划或响应事件运行它是另一回事。工作流编排工具（如Prefect）旨在管理具有多个步骤、依赖关系和调度需求的复杂管道。

    ### 为什么需要编排？

    在生产环境中，ML管道需要：

    - **定时执行**：每天、每小时或按需运行
    - **依赖管理**：确保任务按正确顺序执行
    - **错误处理**：自动重试失败的任务
    - **监控**：跟踪执行状态和性能
    - **可观测性**：日志、指标、告警

    ### 编排工具的作用

    - 调度和自动化管道
    - 管理任务依赖关系
    - 处理失败和重试
    - 提供监控和可观测性
    - 支持多种执行环境
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🎯 Prefect：现代工作流编排

    Prefect是一个开源的编排工具（也有Cloud/Enterprise版本），设计得比旧的基于DAG的系统（如Airflow）更Pythonic和灵活。

    ### 核心概念

    #### 1. Flows & Tasks

    - **Task**：使用`@task`装饰器将Python函数转换为任务
    - **Flow**：使用`@flow`装饰器将任务组合成工作流
    - **依赖推断**：从函数调用方式自动推断依赖关系

    #### 2. 动态工作流

    - 可以在flow内使用标准Python控制流（`if`、`for`等）
    - 不需要显式定义DAG
    - 更灵活、更易于理解

    #### 3. 执行后端

    - 本地执行
    - Dask分布式执行
    - Docker容器
    - Kubernetes
    - 云平台（AWS、GCP、Azure）

    #### 4. Agent/Worker模型

    - Worker轮询工作池获取调度的flow运行
    - 可以在不同环境中运行flow
    - 无需保持终端打开

    ### DAG（有向无环图）

    **什么是DAG？**

    - **节点**：任务（如获取数据、处理数据、训练模型）
    - **边**：依赖关系（如训练依赖于处理，处理依赖于获取）
    - **有向**：任务必须按定义的顺序执行
    - **无环**：没有循环，避免无限运行

    **Prefect vs Airflow：**

    - Airflow：显式定义DAG
    - Prefect：从Python代码隐式构建DAG
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 💻 Prefect示例：简单的ML管道

    让我们通过一个简单的示例来理解Prefect的工作方式。
    """
    )
    return


@app.cell
def _():
    print("💻 Prefect工作流示例")
    print("=" * 50)

    prefect_example = """
from prefect import task, flow
import time

@task(retries=2, retry_delay_seconds=5)
def fetch_data():
    '''模拟从数据源获取数据'''
    print("📥 正在获取数据...")
    time.sleep(1)
    # 模拟数据
    data = {"records": 1000, "features": 10}
    print(f"✅ 获取了 {data['records']} 条记录")
    return data

@task(retries=2, retry_delay_seconds=5)
def process_data(raw_data):
    '''模拟数据处理'''
    print("🔄 正在处理数据...")
    time.sleep(1)
    processed = {
        "records": raw_data["records"],
        "features": raw_data["features"] + 5  # 添加了5个新特征
    }
    print(f"✅ 处理完成，现在有 {processed['features']} 个特征")
    return processed

@task(retries=2, retry_delay_seconds=5)
def train_model(processed_data):
    '''模拟模型训练'''
    print("🎓 正在训练模型...")
    time.sleep(2)
    accuracy = 0.85
    print(f"✅ 模型训练完成，准确率: {accuracy:.2%}")
    return {"accuracy": accuracy, "model_id": "model_v1"}

@flow(name="ml-pipeline")
def ml_pipeline():
    '''完整的ML管道'''
    print("🚀 开始ML管道执行")

    # 依赖关系自动推断
    raw = fetch_data()
    processed = process_data(raw)
    model = train_model(processed)

    print(f"🎉 管道执行完成！模型ID: {model['model_id']}")
    return model

# 运行flow
if __name__ == "__main__":
    result = ml_pipeline()
    """

    print("\n📝 Prefect代码示例：")
    print(prefect_example)

    print("\n🔍 关键特性：")
    print("   1. @task装饰器：使函数可重试、可监控、可观测")
    print("   2. @flow装饰器：定义工作流")
    print("   3. 依赖推断：从函数调用自动推断依赖关系")
    print("   4. 重试机制：自动处理瞬时失败")
    print("   5. 日志记录：自动记录执行日志")

    return (prefect_example,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📅 Prefect调度

    Prefect支持多种调度方式：

    ### 1. Cron调度

    传统的基于时间的调度：

    ```yaml
    schedules:
      - cron: "0 2 * * *"  # 每天凌晨2点
        timezone: "Asia/Shanghai"
    ```

    **常用Cron表达式：**
    - `0 * * * *`：每小时
    - `0 0 * * *`：每天午夜
    - `0 0 * * 1`：每周一
    - `0 0 1 * *`：每月1号

    ### 2. 间隔调度

    按固定间隔运行：

    ```yaml
    schedules:
      - interval: 30  # 每30秒
    ```

    **适用场景：**
    - 实时数据处理
    - 频繁的数据同步
    - 监控任务

    ### 3. 事件驱动触发

    响应事件运行：

    - 文件到达S3
    - Webhook触发
    - 上游任务完成
    - 自定义事件

    **优势：**
    - 不需要轮询
    - 按需运行
    - 节省资源
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🎛️ Prefect部署配置

    ### prefect.yaml示例

    ```yaml
    name: ml-pipeline-project

    deployments:
      - name: ml-pipeline-deployment
        entrypoint: pipeline.py:ml_pipeline
        work_pool:
          name: default-agent-pool
        schedules:
          - interval: 30
            timezone: "Asia/Shanghai"
    ```

    ### 运行Prefect Flow

    #### 开发模式

    ```bash
    # 直接运行（立即执行）
    python pipeline.py
    ```

    #### 生产模式

    ```bash
    # 1. 启动Prefect服务器
    prefect server start

    # 2. 创建部署
    prefect deploy

    # 3. 启动worker
    prefect worker start --pool default-agent-pool
    ```

    ### Prefect UI

    Prefect提供了一个现代化的Web UI：

    - 查看flow运行历史
    - 监控任务状态
    - 查看日志
    - 管理调度
    - 配置告警

    **访问方式：**
    - 本地：`http://localhost:4200`
    - Prefect Cloud：托管服务
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🏆 调度最佳实践

    无论使用哪个编排工具（Airflow、Prefect等），以下最佳实践都适用：

    ### 1. 明智使用Cron/时间调度

    - **留出缓冲时间**：如果数据在凌晨1点可用，调度在3点运行
    - **考虑时区**：明确指定时区，避免混淆
    - **避免高峰期**：不要在系统负载高的时候调度

    ### 2. 事件驱动触发

    - **按需运行**：只在需要时运行，节省资源
    - **减少轮询**：使用事件通知而不是定期检查
    - **集成外部系统**：云存储通知、Webhook等

    ### 3. 重试和幂等性

    - **配置重试**：为易失败的任务设置重试
    - **幂等性**：确保任务可以安全地重复运行
    - **避免副作用**：失败重试不应破坏数据

    **幂等性示例：**

    ```python
    # ❌ 非幂等
    def append_data(data):
        existing = load_data()
        existing.append(data)  # 重试会重复添加
        save_data(existing)

    # ✅ 幂等
    def upsert_data(data):
        existing = load_data()
        existing[data.id] = data  # 重试只是覆盖
        save_data(existing)
    ```

    ### 4. 环境隔离

    - **开发/测试/生产分离**：使用不同的项目或命名空间
    - **避免冲突**：测试不应影响生产
    - **配置管理**：使用环境变量管理配置

    ### 5. 容器化

    - **Docker容器**：确保环境一致性
    - **版本锁定**：固定依赖版本
    - **避免"在我机器上能运行"**：容器化消除环境差异

    ### 6. 文档化

    - **记录管道用途**：每个管道做什么
    - **记录调度**：何时运行、为什么
    - **记录依赖**：上游/下游数据
    - **使用描述**：在编排工具中添加描述
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📊 Spark vs Pandas vs Prefect：工具对比

    让我们通过一个表格来总结这些工具的特点和使用场景：
    """
    )
    return


@app.cell
def _(mo, pd):
    tools_comparison = pd.DataFrame([
        {
            "工具": "Pandas",
            "用途": "数据处理",
            "适用场景": "小到中等数据集（<几GB）",
            "优势": "简单易用、快速、丰富API",
            "劣势": "受限于单机内存",
            "典型使用": "探索性分析、原型开发、特征工程"
        },
        {
            "工具": "Spark",
            "用途": "大数据处理",
            "适用场景": "大数据集（>几GB，数十亿行）",
            "优势": "分布式、可扩展、容错",
            "劣势": "启动开销大、调试困难",
            "典型使用": "大规模ETL、分布式训练、数据湖处理"
        },
        {
            "工具": "Prefect",
            "用途": "工作流编排",
            "适用场景": "复杂管道、生产环境",
            "优势": "灵活、Pythonic、易于监控",
            "劣势": "需要额外基础设施",
            "典型使用": "调度ML管道、依赖管理、自动化"
        }
    ])

    mo.md(f"""
    ### 工具对比表

    {tools_comparison.to_markdown(index=False)}

    ### 组合使用

    在实际的MLOps系统中，这些工具通常组合使用：

    1. **Pandas + Prefect**：
       - 小到中等数据集的ML管道
       - Prefect调度和编排
       - Pandas处理数据和特征工程

    2. **Spark + Prefect**：
       - 大数据ML管道
       - Prefect调度和编排
       - Spark处理大规模数据

    3. **Pandas + Spark + Prefect**：
       - 混合管道
       - Spark处理大数据ETL
       - Pandas处理聚合后的数据
       - Prefect编排整个流程
    """)

    return (tools_comparison,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🎯 实践建议

    ### 学习路径

    #### 1. 掌握基础

    - **Pandas**：数据处理基础
    - **Scikit-learn**：机器学习基础
    - **Python**：编程基础

    #### 2. 理解分布式计算

    - **Spark基础**：RDD、DataFrame、Transformations、Actions
    - **Spark SQL**：分布式SQL查询
    - **Spark MLlib**：分布式机器学习

    #### 3. 学习工作流编排

    - **Prefect基础**：Tasks、Flows、Deployments
    - **调度策略**：Cron、Interval、Event-driven
    - **监控和日志**：UI、日志分析

    ### 实践项目

    #### 项目1：小规模ML管道

    - 使用Pandas处理数据
    - 使用Scikit-learn训练模型
    - 使用Prefect调度和监控

    #### 项目2：大规模数据处理

    - 使用Spark处理大数据集
    - 实现分布式特征工程
    - 使用Spark MLlib训练模型

    #### 项目3：端到端MLOps系统

    - Spark进行大规模ETL
    - Pandas进行特征工程
    - Scikit-learn训练模型
    - Prefect编排整个流程
    - Docker容器化
    - 监控和告警

    ### 推荐资源

    #### Spark学习资源

    - **官方文档**：https://spark.apache.org/docs/latest/
    - **PySpark教程**：https://spark.apache.org/docs/latest/api/python/
    - **Spark权威指南**：书籍推荐

    #### Prefect学习资源

    - **官方文档**：https://docs.prefect.io/
    - **教程**：https://docs.prefect.io/tutorials/
    - **社区**：https://discourse.prefect.io/

    #### MLOps学习资源

    - **MLOps社区**：https://mlops.community/
    - **论文和博客**：关注最新研究和实践
    - **开源项目**：参与和学习
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🎓 总结

    在本章中，我们深入探讨了MLOps中的数据和管道工程：

    ### 关键要点

    #### Apache Spark

    1. **分布式计算引擎**：处理超大数据集
    2. **DataFrame API**：类似Pandas但分布式
    3. **MLlib**：分布式机器学习库
    4. **懒执行**：优化执行计划
    5. **容错机制**：自动处理失败

    #### Pandas vs Spark

    1. **Pandas**：小数据、快速、简单
    2. **Spark**：大数据、分布式、可扩展
    3. **选择标准**：数据大小、复杂度、资源
    4. **组合使用**：发挥各自优势

    #### Prefect工作流编排

    1. **Tasks & Flows**：构建管道
    2. **调度策略**：Cron、Interval、Event-driven
    3. **重试机制**：处理失败
    4. **监控UI**：可观测性
    5. **灵活部署**：多种执行环境

    #### 最佳实践

    1. **明智调度**：考虑时区、缓冲时间
    2. **幂等性**：安全重试
    3. **环境隔离**：开发/测试/生产分离
    4. **容器化**：环境一致性
    5. **文档化**：记录所有内容

    ### 下一步

    在接下来的章节中，我们将继续探讨：

    - **模型开发和实践**：训练、评估、优化
    - **CI/CD工作流**：为ML系统定制
    - **真实案例研究**：行业实践
    - **监控和观测**：生产环境
    - **LLMOps特殊考虑**：大语言模型运维

    ### 核心理念

    MLOps不仅仅是工具和技术，更是一种思维方式：

    - **系统思维**：将ML视为软件系统的一部分
    - **自动化优先**：减少手动操作
    - **可观测性**：了解系统状态
    - **持续改进**：迭代优化
    - **团队协作**：跨职能合作

    通过结合扎实的数据工程（ETL、采样、避免泄漏）、健壮的管道和工作流编排，你可以构建可扩展、可靠的ML系统。

    记住：**深入理解底层系统设计和生命周期原则，你就能够驾驭任何技术栈。** 🚀
    """
    )
    return


@app.cell
def _():
    from datetime import datetime
    return (datetime,)


if __name__ == "__main__":
    app.run()

