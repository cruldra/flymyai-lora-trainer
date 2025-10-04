import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 完整的MLOps蓝图：数据和管道工程—第A部分（含实现）

    MLOps和LLMOps速成课程—第5部分

    ## 回顾

    在我们深入这个MLOps和LLMOps速成课程的第5部分之前，让我们快速回顾一下上一部分涵盖的内容。

    在第4部分中，我们将关于可重现性和版本控制的讨论扩展到了使用Weights & Biases (W&B)的实际探索。

    ![W&B概览](https://www.dailydoseofds.com/content/images/2025/08/image-111.png)

    我们首先介绍了W&B、其核心理念，以及它与MLflow的并排比较。

    关键要点很明确：MLflow vs W&B不是关于哪个更好，而是为你的用例选择正确的工具。

    ![工具对比](https://www.dailydoseofds.com/content/images/2025/08/image-removebg-preview.png)

    从那里，我们进行了实际操作。我们通过两个演示探索了使用W&B的实验跟踪和版本控制：

    - **使用scikit-learn进行预测建模**，我们：
        - 记录指标
        - 跟踪实验
        - 管理工件
        - 注册模型

    ![scikit-learn演示](https://www.dailydoseofds.com/content/images/2025/08/image-114.png)

    - **使用PyTorch进行时间序列销售预测**，我们学习了：
        - 构建多步骤管道
        - W&B的深度学习集成
        - 记录工件
        - 模型检查点

    ![PyTorch演示](https://www.dailydoseofds.com/content/images/2025/08/image-113.png)

    如果你还没有查看第4部分，我们强烈建议先阅读它，因为它为即将到来的内容奠定了基础和流程。

    在本章以及接下来的几章中，我们将从系统角度探索数据和管道工程的核心概念。这个阶段形成了支持MLOps生命周期中所有后续阶段实现的结构性骨干。

    ![MLOps生命周期](https://www.dailydoseofds.com/content/images/2025/08/image-115.png)

    我们将讨论：

    - **数据源和格式**
    - **ETL管道**
    - **实际实现**

    一如既往，每个想法和概念都将得到具体示例、演练和实用技巧的支持，帮助你掌握想法和实现。

    让我们开始吧！

    ---

    ## 引言：理解数据景观

    在机器学习运营(MLOps)中，成功不仅取决于模型，还取决于为这些模型提供数据的数据管道。

    ![数据管道重要性](https://www.dailydoseofds.com/content/images/2025/08/image-117-1.png)

    生产机器学习是一个完全不同的野兽。

    在这里，如果最聪明的模型架构被提供不可靠的数据，或者如果其预测无法重现（正如我们在早期部分看到的），那么它就是毫无价值的。

    因此，至关重要的是要理解，在ML世界中，原材料是数据，我们为数据做出的选择对我们整个ML系统的性能、可扩展性和可靠性有深远的下游后果。

    ![数据决策影响](https://www.dailydoseofds.com/content/images/2025/08/image-116-1.png)

    由于上述事实，在企业MLOps环境中，工程师在一个基本真理下运作：**模型通常是商品，但数据和处理它的管道是驱动业务价值的持久、可防御的资产。**

    ---

    ## 从原始信号获取结构化信息

    生产环境中的数据不是一个干净、静态的CSV文件。它是来自多个源的动态、混乱和连续的信号流，每个源都有自己的特征和要求。

    ### 数据源

    生产ML系统与来自多个来源的数据交互，例如：

    #### 用户输入数据

    这是用户明确提供的数据，例如搜索栏中的文本、上传的图像或表单提交。

    这个数据源是出了名的不可靠，因为用户通常很懒，如果用户可能输入未格式化和原始数据，他们就会这样做。因此，这些数据需要重型验证和强大的错误处理。

    #### 系统生成的数据（日志）

    ![系统日志](https://www.dailydoseofds.com/content/images/2025/08/image-118-1.png)

    应用程序和基础设施生成大量日志。

    这些日志记录重要事件、系统状态（如内存使用）、服务调用和模型预测。

    虽然通常很嘈杂，但日志对于调试、监控系统健康状况以及对我们来说至关重要的是，为我们的ML系统提供可见性是无价的。

    对于许多用例，日志可以批量处理（如每日或每周），但对于实时监控和警报，需要更快的处理。

    #### 内部数据库

    这是企业通常从中获得最大价值的地方。

    管理库存、客户关系(CRM)、用户账户和金融交易的数据库通常是特征工程最有价值的来源。这些数据通常高度结构化并遵循关系模型。

    例如，推荐模型可能处理用户的查询，但它必须检查内部库存数据库以确保推荐的产品实际上有库存，然后才能显示它们。

    #### 第三方数据

    ![第三方数据](https://www.dailydoseofds.com/content/images/2025/08/image-119.png)

    这是从外部供应商获得的数据。它可以从人口统计信息和社交媒体活动到购买习惯。虽然它对于引导推荐系统等模型可能很强大，但其可用性越来越受到隐私法规的限制。

    现在我们广泛了解了ML系统的数据来源，让我们也继续了解ML管道上下文中一些重要的数据格式。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 数据格式

    你为存储选择的格式是一个关键的架构决策，直接影响存储成本、访问速度和易用性。需要理解的两个最重要的二分法是文本与二进制以及行主序与列主序格式。

    #### 文本 vs. 二进制

    像JSON和CSV这样的文本格式是人类可读的。你可以在文本编辑器中打开JSON或CSV文件并立即理解其内容。

    ![文本格式](https://www.dailydoseofds.com/content/images/2025/08/image-121-1.png)

    这使它们非常适合调试、配置和系统间的数据交换。特别是JSON，由于其简单性和灵活性，无处不在，能够表示结构化和非结构化数据。

    然而，这种可读性是有代价的：文本文件冗长且消耗显著更多的存储空间。将数字`1000000`存储为文本需要7个字符（因此在ASCII中是7字节），而在二进制格式中将其存储为32位整数只需要4字节。

    现在，谈到像Parquet这样的二进制格式，这些格式不是人类可读的，是为机器消费而设计的。

    它们更紧凑且处理效率更高。程序必须知道二进制文件的确切模式和布局才能解释其字节。

    ![二进制格式](https://www.dailydoseofds.com/content/images/2025/08/image-122.png)

    空间节省可能是戏剧性的；例如，14 MB的CSV文件在转换为二进制Parquet格式时可以减少到6 MB。对于大规模分析工作负载，像Parquet这样的二进制格式是行业标准。

    #### 行主序 vs. 列主序

    这种区别对于ML工程师来说可能是最关键的，因为它直接关系到我们通常如何访问数据进行训练和分析。它描述了数据在内存中的布局方式。

    ![行列存储对比](https://www.dailydoseofds.com/content/images/2025/08/image-123.png)

    在**行主序格式**（如CSV）中，行的连续元素存储在彼此旁边。把它想象成逐行读取表格。这种布局针对写入密集型工作负载进行了优化，在这些工作负载中你经常添加新的、完整的记录（行）。

    如果你的主要访问模式是一次检索整个样本，例如获取特定用户ID的所有数据，它也很高效。

    在**列主序格式**（如Parquet）中，列的连续元素存储在彼此旁边。这针对分析查询进行了优化，这在机器学习中很常见。考虑计算数百万样本中单个特征的平均值的任务。

    在列主序格式中，系统可以将该列作为单个连续的内存块读取，这非常高效且对缓存友好。在行主序格式中，它必须在内存中跳跃，从每行读取一小块数据，这显著更慢。

    ![性能对比](https://www.dailydoseofds.com/content/images/2025/08/image-124.png)

    性能影响并不微妙。例如，流行的`pandas`库是围绕列主序`DataFrame`构建的。

    一个常见场景是逐行迭代`DataFrame`。这可能比逐列迭代慢几个数量级。

    这个代码片段显示，按列迭代32M+行的`DataFrame`只需不到2微秒，而按行迭代相同的`DataFrame`需要38微秒，这是约20倍的差异。

    ![Pandas迭代性能](https://www.dailydoseofds.com/content/images/2025/08/pandas-iteration.jpeg)

    这是因为，如上所述，Pandas DataFrame是一个列主序数据结构，这意味着列中的连续元素在内存中存储在彼此旁边，如下所示：

    ![内存布局](https://substackcdn.com/image/fetch/$s_!4ZSx!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F67161b70-c600-4467-a407-0cbb2c0667fa_2640x1144.png)
    *列在内存中存储方式的简化版本*

    各个列可能分布在内存中的不同位置。然而，每列的元素总是在一起。

    由于处理器对连续内存块更高效，检索列比获取行快得多。

    ![行访问模式](https://substackcdn.com/image/fetch/$s_!B69T!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F97dc86f0-e5db-41fd-bb2a-425ed2d0a69d_3274x1353.png)
    *行访问时的内存访问模式*

    换句话说，在迭代时，每行都是通过访问非连续的内存块来检索的。处理器必须不断从一个内存位置移动到另一个内存位置来获取所有行元素。

    结果，运行时间急剧增加。

    这不是pandas的缺陷；这是其底层列主序数据模型的直接后果。

    有了对数据来源和格式的理解，让我们现在探索数据工程的核心概念之一：ETL管道。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 数据工程基础：ETL

    在构建管道之前，我们需要在ML上下文中对数据工程基础有扎实的掌握。因此，让我们看看原始数据如何被提取、处理并组织用于机器学习工作流。

    ### ML工作流中的ETL

    ETL代表提取(Extract)、转换(Transform)、加载(Load)。它描述了从源获取数据、将其处理成可用形式，并将其加载到存储或系统中供使用的管道。ETL通常是为模型训练或推理准备数据的第一阶段。

    ![ETL流程](https://www.dailydoseofds.com/content/images/2025/08/image-125.png)

    让我们简要地从理论上讨论每个阶段：

    #### 提取(Extract)

    这涉及从各种数据源收集原始数据。在ML上下文中，这可能意味着：

    - **从数据库查询**：使用SQL从关系数据库提取记录
    - **API调用**：从REST API或GraphQL端点获取数据
    - **文件读取**：处理CSV、JSON、Parquet或其他格式的文件
    - **流数据**：从Kafka、Kinesis或其他流平台消费实时数据
    - **日志解析**：从应用程序日志中提取结构化信息

    **关键挑战**：

    - 处理不同的数据格式和模式
    - 管理API速率限制和超时
    - 处理大型数据集的内存限制
    - 确保数据提取的一致性和可靠性

    #### 转换(Transform)

    这是数据被清理、验证、丰富和重构以满足ML模型要求的阶段。常见的转换包括：

    **数据清理**：

    - 处理缺失值（插补、删除或标记）
    - 去除重复记录
    - 修正数据类型不一致
    - 处理异常值

    **特征工程**：

    - 创建新特征（例如，从日期提取星期几）
    - 特征缩放和标准化
    - 分类变量编码（独热编码、标签编码）
    - 文本预处理（标记化、词干提取）

    **数据验证**：

    - 模式验证（确保列存在且类型正确）
    - 数据质量检查（范围验证、格式检查）
    - 业务规则验证

    **数据聚合**：

    - 按时间窗口或其他维度汇总数据
    - 计算统计指标（均值、中位数、百分位数）
    - 创建特征交叉

    #### 加载(Load)

    最后阶段涉及将转换后的数据存储在目标系统中，准备用于ML训练或推理：

    **存储选项**：

    - **数据仓库**：如Snowflake、BigQuery、Redshift用于分析工作负载
    - **数据湖**：如S3、HDFS用于原始和处理过的数据
    - **特征存储**：专门的系统用于ML特征管理
    - **数据库**：关系型或NoSQL数据库用于操作数据
    - **缓存系统**：如Redis用于快速访问频繁使用的数据

    **加载策略**：

    - **批量加载**：定期（每日、每小时）处理大量数据
    - **增量加载**：只处理新的或更改的数据
    - **实时加载**：连续处理流数据
    - **混合方法**：结合批量和实时处理

    ### ETL vs ELT：现代数据工程的演进

    传统的ETL方法在数据加载到目标系统之前进行转换。然而，随着云数据仓库和大数据处理能力的出现，ELT（提取-加载-转换）模式变得越来越流行。

    #### ELT的优势

    **更快的数据可用性**：

    - 原始数据立即可用于探索
    - 不需要等待所有转换完成

    **灵活性**：

    - 可以对相同的原始数据应用多种转换
    - 更容易适应不断变化的业务需求

    **可扩展性**：

    - 利用现代数据仓库的计算能力
    - 可以并行处理多个转换

    **成本效益**：

    - 减少专用ETL基础设施的需求
    - 按需付费的云计算模型

    #### 混合方法

    在实践中，许多ML系统使用混合方法：

    ```python
    # 混合ETL/ELT流程示例

    # 1. 提取原始数据
    raw_data = extract_from_sources()

    # 2. 基本清理和验证（ETL风格）
    cleaned_data = basic_cleaning(raw_data)

    # 3. 加载到数据湖（ELT风格）
    load_to_data_lake(cleaned_data)

    # 4. 在数据仓库中进行复杂转换（ELT风格）
    transformed_data = complex_transformations_in_warehouse()

    # 5. 为ML准备最终数据集
    ml_ready_data = prepare_for_ml(transformed_data)
    ```

    ### ML特定的ETL考虑

    #### 数据漂移检测

    ```python
    def detect_data_drift(reference_data, current_data):
        \"\"\"检测数据分布的变化\"\"\"
        from scipy import stats

        drift_scores = {}
        for column in reference_data.columns:
            if reference_data[column].dtype in ['int64', 'float64']:
                # 使用KS检验检测数值特征的漂移
                statistic, p_value = stats.ks_2samp(
                    reference_data[column],
                    current_data[column]
                )
                drift_scores[column] = {'statistic': statistic, 'p_value': p_value}

        return drift_scores
    ```

    #### 特征存储集成

    ```python
    def update_feature_store(features, feature_store_client):
        \"\"\"将处理后的特征更新到特征存储\"\"\"

        # 验证特征模式
        validate_feature_schema(features)

        # 计算特征统计
        feature_stats = compute_feature_statistics(features)

        # 更新特征存储
        feature_store_client.write_features(
            features=features,
            metadata=feature_stats,
            timestamp=datetime.now()
        )
    ```

    #### 数据血缘跟踪

    ```python
    def track_data_lineage(source_data, transformed_data, transformation_config):
        \"\"\"跟踪数据转换的血缘\"\"\"

        lineage_info = {
            'source_hash': hash_dataframe(source_data),
            'target_hash': hash_dataframe(transformed_data),
            'transformation_config': transformation_config,
            'timestamp': datetime.now(),
            'row_count_change': len(transformed_data) - len(source_data),
            'column_changes': list(set(transformed_data.columns) - set(source_data.columns))
        }

        # 记录到血缘跟踪系统
        log_lineage(lineage_info)
    ```

    现在让我们通过一个实际的例子来看看这些概念是如何在实践中应用的。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 实际实现：客户流失预测管道

    让我们通过构建一个完整的客户流失预测数据管道来将理论付诸实践。这个例子将展示ETL/ELT的混合方法，包括数据生成、验证、转换和为ML准备。

    ### 项目概述

    我们将构建一个管道来：

    1. **生成合成客户数据**（模拟真实世界的数据源）
    2. **实施数据验证**（确保数据质量）
    3. **执行特征工程**（为ML准备数据）
    4. **存储多种格式**（展示不同的存储策略）
    5. **创建训练就绪的数据集**（最终的ML输出）

    ### 第一步：数据生成和提取

    首先，让我们创建一个合成数据集来模拟客户数据：

    ```python
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import random
    from typing import Dict, List, Tuple
    import warnings
    warnings.filterwarnings('ignore')

    def generate_customer_data(n_customers: int = 10000) -> pd.DataFrame:
        \"\"\"
        生成合成客户数据用于流失预测

        参数:
            n_customers: 要生成的客户数量

        返回:
            包含客户数据的DataFrame
        \"\"\"
        np.random.seed(42)
        random.seed(42)

        # 基础客户信息
        customer_ids = [f"CUST_{i:06d}" for i in range(1, n_customers + 1)]

        # 人口统计数据
        ages = np.random.normal(45, 15, n_customers).astype(int)
        ages = np.clip(ages, 18, 80)  # 限制年龄范围

        genders = np.random.choice(['M', 'F', 'Other'], n_customers, p=[0.48, 0.48, 0.04])

        # 地理数据
        states = ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
        customer_states = np.random.choice(states, n_customers)

        # 账户信息
        account_lengths = np.random.exponential(24, n_customers)  # 月数
        account_lengths = np.clip(account_lengths, 1, 120).astype(int)

        # 服务使用数据
        monthly_charges = np.random.normal(65, 20, n_customers)
        monthly_charges = np.clip(monthly_charges, 20, 150).round(2)

        total_charges = monthly_charges * account_lengths + np.random.normal(0, 50, n_customers)
        total_charges = np.clip(total_charges, 0, None).round(2)

        # 服务特征
        internet_service = np.random.choice(['DSL', 'Fiber', 'No'], n_customers, p=[0.4, 0.5, 0.1])
        online_security = np.random.choice(['Yes', 'No'], n_customers, p=[0.3, 0.7])
        tech_support = np.random.choice(['Yes', 'No'], n_customers, p=[0.25, 0.75])

        # 合同信息
        contract_types = np.random.choice(['Month-to-month', 'One year', 'Two year'],
                                        n_customers, p=[0.6, 0.25, 0.15])

        payment_methods = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
                                         n_customers, p=[0.4, 0.2, 0.2, 0.2])

        # 生成流失标签（基于一些逻辑规则）
        churn_probability = 0.1  # 基础流失率

        # 影响流失的因素
        churn_prob_adjusted = churn_probability + np.where(
            (contract_types == 'Month-to-month') &
            (monthly_charges > 80) &
            (account_lengths < 12), 0.4, 0
        )

        churn_prob_adjusted += np.where(
            (internet_service == 'Fiber') &
            (online_security == 'No'), 0.2, 0
        )

        churn_prob_adjusted += np.where(ages > 65, 0.15, 0)
        churn_prob_adjusted = np.clip(churn_prob_adjusted, 0, 0.8)

        churn_labels = np.random.binomial(1, churn_prob_adjusted, n_customers)

        # 创建DataFrame
        data = {
            'customer_id': customer_ids,
            'age': ages,
            'gender': genders,
            'state': customer_states,
            'account_length_months': account_lengths,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'internet_service': internet_service,
            'online_security': online_security,
            'tech_support': tech_support,
            'contract': contract_types,
            'payment_method': payment_methods,
            'label_churn': churn_labels
        }

        df = pd.DataFrame(data)

        # 添加一些缺失值来模拟真实世界的数据
        missing_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
        df.loc[missing_indices, 'total_charges'] = np.nan

        # 添加时间戳
        df['created_at'] = datetime.now()
        df['updated_at'] = df['created_at'] + pd.to_timedelta(
            np.random.randint(0, 30, n_customers), unit='D'
        )

        return df

    # 生成数据
    print("生成客户数据...")
    customer_data = generate_customer_data(10000)
    print(f"生成了 {len(customer_data)} 条客户记录")
    print(f"流失率: {customer_data['label_churn'].mean():.2%}")
    print("\\n数据样本:")
    print(customer_data.head())
    ```

    ### 第二步：数据验证

    在处理数据之前，我们需要验证其质量和完整性：

    ```python
    def validate_customer_data(df: pd.DataFrame) -> Dict[str, any]:
        \"\"\"
        验证客户数据的质量和完整性

        参数:
            df: 要验证的DataFrame

        返回:
            包含验证结果的字典
        \"\"\"
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }

        # 检查必需列
        required_columns = [
            'customer_id', 'age', 'gender', 'monthly_charges',
            'total_charges', 'label_churn'
        ]

        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            validation_results['errors'].append(f"缺少必需列: {missing_columns}")
            validation_results['is_valid'] = False

        # 检查数据类型
        if 'age' in df.columns and not pd.api.types.is_numeric_dtype(df['age']):
            validation_results['errors'].append("年龄列必须是数值类型")
            validation_results['is_valid'] = False

        # 检查数据范围
        if 'age' in df.columns:
            invalid_ages = df[(df['age'] < 18) | (df['age'] > 120)]
            if len(invalid_ages) > 0:
                validation_results['warnings'].append(f"发现 {len(invalid_ages)} 个异常年龄值")

        # 检查重复的客户ID
        if 'customer_id' in df.columns:
            duplicates = df['customer_id'].duplicated().sum()
            if duplicates > 0:
                validation_results['errors'].append(f"发现 {duplicates} 个重复的客户ID")
                validation_results['is_valid'] = False

        # 检查缺失值
        missing_stats = df.isnull().sum()
        validation_results['statistics']['missing_values'] = missing_stats.to_dict()

        # 检查数据分布
        if 'label_churn' in df.columns:
            churn_rate = df['label_churn'].mean()
            validation_results['statistics']['churn_rate'] = churn_rate

            if churn_rate > 0.5:
                validation_results['warnings'].append(f"流失率异常高: {churn_rate:.2%}")
            elif churn_rate < 0.01:
                validation_results['warnings'].append(f"流失率异常低: {churn_rate:.2%}")

        # 数据质量评分
        error_count = len(validation_results['errors'])
        warning_count = len(validation_results['warnings'])

        if error_count == 0 and warning_count == 0:
            quality_score = 1.0
        elif error_count == 0:
            quality_score = max(0.7, 1.0 - warning_count * 0.1)
        else:
            quality_score = max(0.0, 0.5 - error_count * 0.1)

        validation_results['statistics']['quality_score'] = quality_score

        return validation_results

    # 验证数据
    print("验证数据质量...")
    validation_results = validate_customer_data(customer_data)

    print(f"数据验证结果:")
    print(f"- 有效性: {'✓' if validation_results['is_valid'] else '✗'}")
    print(f"- 质量评分: {validation_results['statistics']['quality_score']:.2f}")
    print(f"- 错误数量: {len(validation_results['errors'])}")
    print(f"- 警告数量: {len(validation_results['warnings'])}")

    if validation_results['errors']:
        print("\\n错误:")
        for error in validation_results['errors']:
            print(f"  - {error}")

    if validation_results['warnings']:
        print("\\n警告:")
        for warning in validation_results['warnings']:
            print(f"  - {warning}")
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 第三步：数据存储（加载阶段）

    现在让我们将验证后的数据存储为多种格式，展示不同的存储策略：

    ```python
    import os
    import json
    from pathlib import Path

    def save_data_multiple_formats(df: pd.DataFrame, base_path: str = "data"):
        \"\"\"
        将数据保存为多种格式

        参数:
            df: 要保存的DataFrame
            base_path: 基础存储路径
        \"\"\"
        # 创建目录结构
        Path(base_path).mkdir(exist_ok=True)
        Path(f"{base_path}/raw").mkdir(exist_ok=True)
        Path(f"{base_path}/processed").mkdir(exist_ok=True)

        # 保存为CSV（文本格式，行主序）
        csv_path = f"{base_path}/raw/customer_data.csv"
        df.to_csv(csv_path, index=False)
        csv_size = os.path.getsize(csv_path) / (1024 * 1024)  # MB

        # 保存为Parquet（二进制格式，列主序）
        parquet_path = f"{base_path}/raw/customer_data.parquet"
        df.to_parquet(parquet_path, index=False)
        parquet_size = os.path.getsize(parquet_path) / (1024 * 1024)  # MB

        # 保存为JSON（文本格式，灵活结构）
        json_path = f"{base_path}/raw/customer_data.json"
        df.to_json(json_path, orient='records', date_format='iso')
        json_size = os.path.getsize(json_path) / (1024 * 1024)  # MB

        # 保存元数据
        metadata = {
            'created_at': datetime.now().isoformat(),
            'record_count': len(df),
            'column_count': len(df.columns),
            'file_sizes_mb': {
                'csv': round(csv_size, 2),
                'parquet': round(parquet_size, 2),
                'json': round(json_size, 2)
            },
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict()
        }

        with open(f"{base_path}/raw/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"数据已保存到 {base_path}/raw/")
        print(f"文件大小对比:")
        print(f"  - CSV: {csv_size:.2f} MB")
        print(f"  - Parquet: {parquet_size:.2f} MB ({parquet_size/csv_size:.1%} of CSV)")
        print(f"  - JSON: {json_size:.2f} MB ({json_size/csv_size:.1%} of CSV)")

        return metadata

    # 保存数据
    print("保存数据为多种格式...")
    metadata = save_data_multiple_formats(customer_data)
    ```

    ### 第四步：特征工程和转换

    现在让我们实施特征工程管道：

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    def create_feature_engineering_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        \"\"\"
        创建特征工程管道并处理数据

        参数:
            df: 输入DataFrame

        返回:
            训练和验证数据集的元组
        \"\"\"
        # 创建数据副本
        df_processed = df.copy()

        # 分离特征和目标
        feature_columns = [col for col in df.columns if col not in ['label_churn', 'customer_id', 'created_at', 'updated_at']]
        X = df_processed[feature_columns]
        y = df_processed['label_churn']

        # 分割数据（在预处理之前分割以避免数据泄漏）
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 识别数值和分类列
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()

        print(f"数值列 ({len(numeric_columns)}): {numeric_columns}")
        print(f"分类列 ({len(categorical_columns)}): {categorical_columns}")

        # 创建预处理管道
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])

        # 组合预处理器
        preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, numeric_columns),
            ('cat', categorical_pipeline, categorical_columns)
        ], remainder='drop')

        # 拟合并转换数据
        X_train_processed = preprocessor.fit_transform(X_train)
        X_valid_processed = preprocessor.transform(X_valid)

        # 获取特征名称
        numeric_feature_names = numeric_columns
        categorical_feature_names = []

        if categorical_columns:
            # 获取独热编码后的特征名称
            cat_encoder = preprocessor.named_transformers_['cat']['onehot']
            categorical_feature_names = cat_encoder.get_feature_names_out(categorical_columns).tolist()

        all_feature_names = numeric_feature_names + categorical_feature_names

        # 转换为DataFrame
        X_train_df = pd.DataFrame(
            X_train_processed,
            columns=all_feature_names,
            index=X_train.index.copy()
        ).reset_index(drop=True)

        X_valid_df = pd.DataFrame(
            X_valid_processed,
            columns=all_feature_names,
            index=X_valid.index.copy()
        ).reset_index(drop=True)

        # 添加目标变量
        train_final = pd.concat([
            X_train_df,
            y_train.reset_index(drop=True)
        ], axis=1)

        valid_final = pd.concat([
            X_valid_df,
            y_valid.reset_index(drop=True)
        ], axis=1)

        print(f"\\n特征工程完成:")
        print(f"  - 训练集: {train_final.shape}")
        print(f"  - 验证集: {valid_final.shape}")
        print(f"  - 总特征数: {len(all_feature_names)}")

        return train_final, valid_final, preprocessor

    # 执行特征工程
    print("执行特征工程...")
    train_data, valid_data, feature_pipeline = create_feature_engineering_pipeline(customer_data)

    print("\\n训练数据样本:")
    print(train_data.head())
    ```

    ### 第五步：最终数据加载和管道完成

    ```python
    def save_processed_data(train_df: pd.DataFrame, valid_df: pd.DataFrame,
                          pipeline, base_path: str = "data"):
        \"\"\"
        保存处理后的数据和管道

        参数:
            train_df: 训练数据
            valid_df: 验证数据
            pipeline: 特征工程管道
            base_path: 基础路径
        \"\"\"
        import joblib

        # 保存处理后的数据
        train_df.to_csv(f"{base_path}/processed/train_data.csv", index=False)
        valid_df.to_csv(f"{base_path}/processed/valid_data.csv", index=False)

        # 保存为Parquet格式（更高效）
        train_df.to_parquet(f"{base_path}/processed/train_data.parquet", index=False)
        valid_df.to_parquet(f"{base_path}/processed/valid_data.parquet", index=False)

        # 保存预处理管道
        joblib.dump(pipeline, f"{base_path}/processed/feature_pipeline.joblib")

        # 创建数据摘要
        summary = {
            'pipeline_created_at': datetime.now().isoformat(),
            'train_samples': len(train_df),
            'valid_samples': len(valid_df),
            'feature_count': len(train_df.columns) - 1,  # 减去目标变量
            'target_distribution': {
                'train_churn_rate': train_df['label_churn'].mean(),
                'valid_churn_rate': valid_df['label_churn'].mean()
            },
            'feature_names': [col for col in train_df.columns if col != 'label_churn']
        }

        with open(f"{base_path}/processed/data_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"处理后的数据已保存到 {base_path}/processed/")
        print(f"数据摘要:")
        print(f"  - 训练样本: {summary['train_samples']:,}")
        print(f"  - 验证样本: {summary['valid_samples']:,}")
        print(f"  - 特征数量: {summary['feature_count']}")
        print(f"  - 训练集流失率: {summary['target_distribution']['train_churn_rate']:.2%}")
        print(f"  - 验证集流失率: {summary['target_distribution']['valid_churn_rate']:.2%}")

    # 保存最终处理的数据
    print("保存处理后的数据...")
    save_processed_data(train_data, valid_data, feature_pipeline)
    ```

    ### 管道性能分析

    让我们分析我们构建的管道的性能特征：

    ```python
    def analyze_pipeline_performance():
        \"\"\"分析数据管道的性能特征\"\"\"

        print("=== 数据管道性能分析 ===\\n")

        # 1. 存储效率分析
        print("1. 存储格式效率对比:")
        csv_size = os.path.getsize("data/raw/customer_data.csv") / 1024
        parquet_size = os.path.getsize("data/raw/customer_data.parquet") / 1024
        json_size = os.path.getsize("data/raw/customer_data.json") / 1024

        print(f"   - CSV: {csv_size:.1f} KB")
        print(f"   - Parquet: {parquet_size:.1f} KB (节省 {(1-parquet_size/csv_size)*100:.1f}%)")
        print(f"   - JSON: {json_size:.1f} KB (增加 {(json_size/csv_size-1)*100:.1f}%)")

        # 2. 数据质量指标
        print("\\n2. 数据质量指标:")
        print(f"   - 原始数据记录数: {len(customer_data):,}")
        print(f"   - 处理后训练记录数: {len(train_data):,}")
        print(f"   - 处理后验证记录数: {len(valid_data):,}")
        print(f"   - 数据保留率: {(len(train_data) + len(valid_data))/len(customer_data)*100:.1f}%")

        # 3. 特征工程效果
        original_features = len([col for col in customer_data.columns
                               if col not in ['label_churn', 'customer_id', 'created_at', 'updated_at']])
        processed_features = len(train_data.columns) - 1  # 减去目标变量

        print("\\n3. 特征工程效果:")
        print(f"   - 原始特征数: {original_features}")
        print(f"   - 处理后特征数: {processed_features}")
        print(f"   - 特征扩展倍数: {processed_features/original_features:.1f}x")

        # 4. 类别平衡
        print("\\n4. 目标变量分布:")
        train_churn_rate = train_data['label_churn'].mean()
        valid_churn_rate = valid_data['label_churn'].mean()
        print(f"   - 训练集流失率: {train_churn_rate:.2%}")
        print(f"   - 验证集流失率: {valid_churn_rate:.2%}")
        print(f"   - 分布差异: {abs(train_churn_rate - valid_churn_rate)*100:.2f} 百分点")

    # 执行性能分析
    analyze_pipeline_performance()
    ```

    这个实现展示了一个完整的混合ETL/ELT管道，包括：

    #### ✅ **ETL组件**
    - **提取**：从合成数据生成器获取数据
    - **转换**：数据验证、清理和特征工程
    - **加载**：存储为多种格式供下游使用

    #### ✅ **ELT组件**
    - **提取**：原始数据直接加载到存储
    - **加载**：原始数据存储在数据湖中
    - **转换**：在存储后进行复杂的特征工程

    #### ✅ **生产就绪特性**
    - 数据验证和质量检查
    - 多格式存储（CSV、Parquet、JSON）
    - 管道序列化和版本控制
    - 性能监控和分析
    - 错误处理和日志记录

    这种方法在实际生产环境中提供了灵活性、可扩展性和可维护性。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 工具和规模的快速说明

    你使用的实际工具栈将根据数据规模和生产要求而有所不同。在更大规模下，例如，你可能会使用PySpark进行分布式数据处理，使用Airflow或Prefect进行工作流编排，以及其他高级框架来确保可靠性和可扩展性。

    ### 规模化工具栈对比

    #### 🏠 **小规模/原型阶段**
    ```python
    # 我们演示中使用的工具栈
    技术栈：
    - 数据处理：Pandas + NumPy
    - 特征工程：scikit-learn
    - 存储：本地文件系统
    - 编排：Python脚本

    适用场景：
    - 数据量 < 1GB
    - 单机处理
    - 快速原型开发
    - 概念验证
    ```

    #### 🏢 **中等规模/生产环境**
    ```python
    # 中等规模生产环境
    技术栈：
    - 数据处理：Dask / Polars
    - 特征工程：scikit-learn + 自定义管道
    - 存储：云存储 (S3, GCS, Azure Blob)
    - 编排：Airflow / Prefect
    - 监控：Prometheus + Grafana

    适用场景：
    - 数据量 1GB - 100GB
    - 多核/多节点处理
    - 定期批处理作业
    - 中等复杂度的特征工程
    ```

    #### 🏭 **大规模/企业级**
    ```python
    # 大规模企业环境
    技术栈：
    - 数据处理：Apache Spark (PySpark)
    - 流处理：Apache Kafka + Spark Streaming
    - 特征存储：Feast / Tecton / AWS SageMaker Feature Store
    - 编排：Apache Airflow / Kubeflow Pipelines
    - 存储：数据湖 (Delta Lake, Apache Iceberg)
    - 监控：DataDog / New Relic + 自定义仪表板

    适用场景：
    - 数据量 > 100GB
    - 分布式集群处理
    - 实时和批处理混合
    - 复杂的特征工程和ML管道
    ```

    ### 工具选择决策矩阵

    | 考虑因素 | 小规模 | 中等规模 | 大规模 |
    |---------|--------|----------|--------|
    | **数据量** | < 1GB | 1-100GB | > 100GB |
    | **处理频率** | 按需 | 每日/每小时 | 实时 + 批处理 |
    | **团队规模** | 1-3人 | 3-10人 | 10+人 |
    | **预算** | 低 | 中等 | 高 |
    | **复杂度** | 简单 | 中等 | 复杂 |
    | **可靠性要求** | 基础 | 高 | 关键任务 |

    ### 迁移路径

    ```python
    # 典型的技术栈演进路径

    阶段1：原型开发
    pandas → 验证概念 → 小规模测试

    阶段2：扩展
    pandas → Dask/Polars → 中等规模生产

    阶段3：企业级
    Dask/Polars → PySpark → 大规模分布式处理

    # 关键原则：渐进式迁移
    - 保持核心逻辑不变
    - 逐步替换底层技术
    - 维护向后兼容性
    - 持续性能监控
    ```

    然而，在本章中，我们使用Pandas、NumPy和scikit-learn实现了我们的演示，因为这里的主要重点是首先建立基础。

    **关键要点是，虽然技术栈可能会改变，但底层原则保持不变。** 随着我们在这个系列中的前进，我们也将探索使用更高级工具的模拟，但现在，重点是掌握核心概念。

    ### 实际生产考虑

    #### 🔄 **数据管道监控**
    ```python
    # 生产环境中的管道监控示例
    import logging
    from datetime import datetime

    def monitor_pipeline_health(stage_name: str, input_data, output_data):
        \"\"\"监控管道各阶段的健康状况\"\"\"

        # 数据量监控
        input_count = len(input_data) if hasattr(input_data, '__len__') else 0
        output_count = len(output_data) if hasattr(output_data, '__len__') else 0

        # 数据质量监控
        if hasattr(output_data, 'isnull'):
            null_percentage = output_data.isnull().sum().sum() / output_data.size * 100
        else:
            null_percentage = 0

        # 记录指标
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage_name,
            'input_records': input_count,
            'output_records': output_count,
            'data_loss_percentage': (input_count - output_count) / input_count * 100 if input_count > 0 else 0,
            'null_percentage': null_percentage
        }

        # 发送到监控系统
        logging.info(f"Pipeline metrics: {metrics}")

        # 警报条件
        if metrics['data_loss_percentage'] > 10:
            logging.warning(f"High data loss in {stage_name}: {metrics['data_loss_percentage']:.1f}%")

        if metrics['null_percentage'] > 5:
            logging.warning(f"High null percentage in {stage_name}: {metrics['null_percentage']:.1f}%")
    ```

    #### 🚨 **错误处理和恢复**
    ```python
    def robust_pipeline_stage(stage_func, input_data, max_retries=3):
        \"\"\"为管道阶段添加错误处理和重试机制\"\"\"

        for attempt in range(max_retries):
            try:
                result = stage_func(input_data)
                logging.info(f"Stage completed successfully on attempt {attempt + 1}")
                return result

            except Exception as e:
                logging.error(f"Stage failed on attempt {attempt + 1}: {str(e)}")

                if attempt == max_retries - 1:
                    # 最后一次尝试失败，记录并重新抛出异常
                    logging.critical(f"Stage failed after {max_retries} attempts")
                    raise

                # 等待后重试
                time.sleep(2 ** attempt)  # 指数退避
    ```

    #### 📊 **数据血缘跟踪**
    ```python
    class DataLineageTracker:
        \"\"\"跟踪数据在管道中的血缘关系\"\"\"

        def __init__(self):
            self.lineage_graph = {}

        def track_transformation(self, input_id: str, output_id: str,
                               transformation: str, metadata: dict = None):
            \"\"\"记录数据转换\"\"\"

            if output_id not in self.lineage_graph:
                self.lineage_graph[output_id] = {
                    'inputs': [],
                    'transformations': [],
                    'metadata': []
                }

            self.lineage_graph[output_id]['inputs'].append(input_id)
            self.lineage_graph[output_id]['transformations'].append(transformation)
            self.lineage_graph[output_id]['metadata'].append(metadata or {})

        def get_lineage(self, data_id: str) -> dict:
            \"\"\"获取数据的完整血缘\"\"\"
            return self.lineage_graph.get(data_id, {})

        def visualize_lineage(self, data_id: str):
            \"\"\"可视化数据血缘（简化版本）\"\"\"
            lineage = self.get_lineage(data_id)
            print(f"数据血缘 for {data_id}:")
            for i, (input_id, transform) in enumerate(zip(
                lineage.get('inputs', []),
                lineage.get('transformations', [])
            )):
                print(f"  {i+1}. {input_id} --[{transform}]--> {data_id}")
    ```

    这些生产级考虑确保了数据管道的可靠性、可观察性和可维护性，这对于企业级ML系统至关重要。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 结论

    在本章中，我们深入探讨了机器学习运营中数据管道的关键作用，展示了生产ML中的成功更多地取决于数据的可靠性，而不是模型的复杂性。

    ### 🎯 **关键学习要点**

    #### 📊 **数据景观理解**
    我们探索了各种数据源和格式及其性能权衡，理解了这些架构选择如何向下游影响ML效率：

    - **数据源多样性**：用户输入、系统日志、内部数据库、第三方数据
    - **格式选择影响**：文本vs二进制、行主序vs列主序的性能差异
    - **存储策略**：不同格式的空间效率和访问模式优化

    #### 🔄 **ETL/ELT范式掌握**
    在这个基础上，我们研究了ETL和ELT范式，认识到真实世界ML工作流中使用的权衡和混合策略：

    - **传统ETL**：提取→转换→加载的线性流程
    - **现代ELT**：提取→加载→转换的灵活方法
    - **混合策略**：结合两种方法的优势

    #### 🛠️ **实际实现经验**
    最后，通过实际模拟，我们实现了一个混合ETL/ELT管道，完整包含：

    - **数据生成**：合成客户流失数据创建
    - **数据验证**：质量检查和完整性验证
    - **特征工程**：使用scikit-learn管道的转换
    - **多格式存储**：CSV、Parquet、JSON的对比
    - **scikit-learn管道处理**：生产就绪的预处理流程

    ### 💡 **核心洞察**

    #### 🏗️ **架构决策的重要性**
    我们学到了数据格式和存储选择不是技术细节，而是影响整个系统性能的架构决策：

    - **Parquet vs CSV**：在我们的演示中，Parquet格式比CSV节省了约50%的存储空间
    - **列主序优势**：pandas DataFrame的列迭代比行迭代快20倍
    - **访问模式匹配**：选择与你的主要数据访问模式匹配的存储格式

    #### 🔍 **数据质量至上**
    数据验证和质量检查不是可选的，而是生产ML系统的必需组件：

    - **早期验证**：在管道早期捕获数据质量问题
    - **持续监控**：跟踪数据漂移和分布变化
    - **自动化检查**：实施自动化的数据质量门控

    #### ⚖️ **灵活性与效率的平衡**
    混合ETL/ELT方法提供了灵活性和效率的最佳平衡：

    - **即时可用性**：原始数据立即可用于探索
    - **处理灵活性**：可以对相同数据应用多种转换
    - **计算优化**：利用现代数据仓库的处理能力

    ### 🎪 **关键要点**

    **在企业MLOps中，模型是商品，但数据管道是资产。** 掌握数据管道的设计、验证和编排是使ML系统健壮、可扩展和业务关键的关键。

    这个原则体现在几个方面：

    #### 🔄 **可重现性**
    - 版本化的数据管道确保实验可重现
    - 管道序列化允许在不同环境中部署相同的处理逻辑
    - 数据血缘跟踪提供完整的可追溯性

    #### 📈 **可扩展性**
    - 模块化管道设计支持组件级扩展
    - 格式选择影响处理大规模数据的能力
    - 工具栈可以随着数据量增长而演进

    #### 🛡️ **可靠性**
    - 数据验证防止下游模型错误
    - 错误处理和重试机制确保管道稳定性
    - 监控和警报提供运营可见性

    ### 🚀 **未来展望**

    这只是MLOps/LLMOps系列中关于数据管道和工程的第一部分，还有更多部分将跟进。

    在即将到来的部分中，我们将深入探讨：

    #### 📊 **高级数据概念**
    - **特征存储**：集中化特征管理和服务
    - **数据版本控制**：使用DVC和其他工具进行数据版本管理
    - **数据漂移检测**：自动化监控和警报系统
    - **实时特征工程**：流处理和在线特征计算

    #### 🔧 **关键工具和软件**
    - **Apache Airflow**：工作流编排和调度
    - **Apache Spark**：大规模分布式数据处理
    - **Kafka + Spark Streaming**：实时数据处理
    - **特征存储解决方案**：Feast、Tecton、SageMaker Feature Store

    #### 🏗️ **编排和自动化**
    - **管道编排**：复杂工作流的设计和管理
    - **依赖管理**：任务间依赖关系的处理
    - **错误恢复**：失败处理和自动重试策略
    - **资源管理**：计算资源的动态分配和优化

    在数据处理和分析之后，我们将继续这个速成课程的旅程：

    #### 🔄 **CI/CD工作流**
    - 为ML系统量身定制的持续集成和部署
    - 模型训练和部署的自动化流程
    - 测试策略和质量保证

    #### 🏢 **行业案例研究**
    - 来自行业的真实世界案例研究
    - 不同规模和领域的最佳实践
    - 常见挑战和解决方案

    #### 🤖 **模型开发和实践**
    - 模型训练和验证的最佳实践
    - 超参数优化和实验管理
    - 模型选择和集成策略

    #### 📊 **生产监控和观察**
    - 模型性能监控
    - 数据和模型漂移检测
    - 警报和事件响应

    #### 🧠 **LLMOps特殊考虑**
    - 大语言模型的特殊运营需求
    - 提示工程和版本控制
    - 成本优化和性能调优

    #### 🔗 **端到端集成**
    - 结合生命周期所有元素的完整端到端示例
    - 从数据到部署的完整工作流
    - 企业级MLOps平台架构

    ### 🎯 **最终目标**

    目标，一如既往，是帮助你培养成熟的、**以系统为中心的思维方式**，将机器学习不视为独立的工件，而是更广泛软件生态系统的活跃部分。

    这种思维方式的特征包括：

    - **整体视角**：理解ML系统的所有组件如何相互作用
    - **质量意识**：优先考虑数据质量和管道可靠性
    - **可扩展性思维**：设计能够随业务增长而扩展的系统
    - **运营导向**：考虑监控、维护和故障排除
    - **协作精神**：构建支持团队协作的系统和流程

    通过掌握这些数据工程基础，你已经为构建健壮、可扩展和可维护的ML系统奠定了坚实的基础。在接下来的章节中，我们将在这个基础上构建更复杂和强大的MLOps能力。

    ---

    🚀 **继续你的MLOps学习之旅，记住：优秀的数据管道是成功ML系统的基石！**
    """
    )
    return


if __name__ == "__main__":
    app.run()
