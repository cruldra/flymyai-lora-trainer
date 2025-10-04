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
    # Feast特征存储完全指南

    ## 🎯 什么是Feast？

    **Feast（Feature Store）** 是一个开源的特征存储系统，专门为机器学习工作流设计。它解决了ML系统中特征管理的核心挑战：**如何在训练和服务之间保持特征的一致性**。

    ### 核心问题

    在传统的ML开发中，我们经常遇到以下问题：

    1. **训练-服务偏差**：训练时的特征计算逻辑与生产服务时不一致
    2. **特征重复开发**：不同团队重复实现相同的特征
    3. **数据泄漏风险**：特征计算中意外使用了未来信息
    4. **特征发现困难**：团队不知道已有哪些特征可用
    5. **实时特征服务**：生产环境需要低延迟的特征查询

    ### Feast的解决方案

    - **统一特征定义**：一次定义，训练和服务都使用相同逻辑
    - **时间点正确性**：确保训练数据不包含未来信息
    - **特征重用**：集中管理，避免重复开发
    - **在线/离线存储**：支持训练（批量）和服务（实时）两种场景
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🏗️ Feast核心概念

    让我们通过交互式代码来理解Feast的核心概念：
    """
    )
    return


@app.cell
def _():
    # 安装必要的包（如果需要）
    import subprocess
    import sys

    def install_package(package):
        try:
            __import__(package)
        except ImportError:
            print(f"安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    # 尝试安装feast（如果未安装）
    try:
        import feast
        print("✅ Feast已安装")
    except ImportError:
        print("⚠️ Feast未安装，请运行: pip install feast")

    # 导入其他必要的库
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import warnings
    warnings.filterwarnings('ignore')

    print("📦 依赖包导入完成")
    return datetime, np, pd, timedelta


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 1. Entity（实体）
    实体是特征的主键，定义了特征属于哪个对象。
    """
    )
    return


@app.cell
def _():
    # 定义客户实体
    from feast import Entity
    from feast.types import ValueType

    customer = Entity(
        name="customer_id",
        value_type=ValueType.STRING,
        description="客户唯一标识符"
    )

    print("✅ 客户实体定义完成")
    print(f"实体名称: {customer.name}")
    print(f"值类型: {customer.value_type}")
    print(f"描述: {customer.description}")
    return (customer,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 2. Feature View（特征视图）
    特征视图定义了一组相关特征及其计算逻辑。
    """
    )
    return


@app.cell
def _(customer, timedelta):
    from feast import FeatureView, Field, FileSource
    from feast.types import Float32, Int64, String

    # 定义数据源
    customer_source = FileSource(
        path="data/customer_features.parquet",
        timestamp_field="event_timestamp"
    )

    # 定义客户人口统计特征视图
    customer_demographics = FeatureView(
        name="customer_demographics",
        entities=[customer],
        ttl=timedelta(days=365),  # 特征的生存时间
        schema=[
            Field(name="age", dtype=Int64),
            Field(name="income", dtype=Float32),
            Field(name="credit_score", dtype=Int64),
        ],
        source=customer_source,
        tags={"team": "ml", "domain": "demographics"}
    )

    # 定义客户行为特征视图
    customer_behavior = FeatureView(
        name="customer_behavior",
        entities=[customer],
        ttl=timedelta(days=90),
        schema=[
            Field(name="monthly_charges", dtype=Float32),
            Field(name="total_charges", dtype=Float32),
            Field(name="support_calls", dtype=Int64),
            Field(name="contract_length", dtype=Int64),
        ],
        source=customer_source,
        tags={"team": "ml", "domain": "behavior"}
    )

    print("✅ 特征视图定义完成")
    print(f"人口统计特征: {[f.name for f in customer_demographics.schema]}")
    print(f"行为特征: {[f.name for f in customer_behavior.schema]}")
    return customer_behavior, customer_demographics, customer_source


@app.cell
def _(customer_behavior, customer_demographics):
    # 定义特征服务
    from feast import FeatureService

    feature_refs = [
        customer_demographics[["age", "income", "credit_score"]],
        customer_behavior[["monthly_charges", "total_charges", "support_calls", "contract_length"]],
    ]

    churn_prediction_service = FeatureService(
        name="churn_prediction",
        features=feature_refs
    )

    print("✅ 特征服务定义完成")
    print(f"服务名称: {churn_prediction_service.name}")
    print(f"包含特征数量: {len(feature_refs)}")
    return churn_prediction_service, feature_refs


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📊 实际案例：客户流失预测

    让我们通过一个完整的客户流失预测案例来演示Feast的使用。
    """
    )
    return


@app.cell
def _(datetime, np, pd, timedelta):
    # 生成示例客户数据
    def generate_customer_data():
        """生成示例客户数据用于演示"""
        np.random.seed(42)
        n_customers = 100  # 为了演示，使用较小的数据集

        # 生成客户基础信息
        customer_data = []
        base_date = datetime(2023, 1, 1)

        for i in range(n_customers):
            customer_id = f"customer_{i:04d}"

            # 生成多个时间点的数据
            for days_offset in [0, 30, 60, 90]:
                event_time = base_date + timedelta(days=days_offset)

                customer_data.append({
                    'customer_id': customer_id,
                    'event_timestamp': event_time,
                    'age': np.random.randint(18, 80),
                    'income': max(20000, np.random.normal(50000, 20000)),
                    'credit_score': np.random.randint(300, 850),
                    'monthly_charges': max(10, np.random.normal(65, 20)),
                    'total_charges': max(0, np.random.normal(1000, 500)),
                    'support_calls': np.random.poisson(2),
                    'contract_length': np.random.choice([1, 12, 24], p=[0.5, 0.3, 0.2])
                })

        df = pd.DataFrame(customer_data)

        # 确保数据类型正确
        df['income'] = df['income'].astype('float32')
        df['monthly_charges'] = df['monthly_charges'].astype('float32')
        df['total_charges'] = df['total_charges'].astype('float32')

        return df

    # 生成数据
    customer_df = generate_customer_data()

    print(f"✅ 生成了 {len(customer_df)} 条客户特征记录")
    print(f"📊 数据形状: {customer_df.shape}")
    print(f"👥 唯一客户数: {customer_df['customer_id'].nunique()}")
    print(f"📅 时间范围: {customer_df['event_timestamp'].min()} 到 {customer_df['event_timestamp'].max()}")

    # 显示数据样本
    print("\n📋 数据样本:")
    customer_df.head()
    return (customer_df,)


@app.cell
def _(customer_df):
    # 数据质量检查
    print("🔍 数据质量检查:")
    print(f"缺失值统计:")
    missing_stats = customer_df.isnull().sum()
    for col, col_missing_count in missing_stats.items():
        if col_missing_count > 0:
            print(f"  {col}: {col_missing_count} ({col_missing_count/len(customer_df)*100:.1f}%)")

    if missing_stats.sum() == 0:
        print("  ✅ 无缺失值")

    print(f"\n📈 数值特征统计:")
    numeric_cols = ['age', 'income', 'credit_score', 'monthly_charges', 'total_charges', 'support_calls']
    customer_df[numeric_cols].describe()
    return missing_stats, numeric_cols


@app.cell
def _(customer_df, mo):
    # 创建数据目录并保存数据
    import os

    # 创建数据目录
    os.makedirs('data', exist_ok=True)

    # 保存数据
    customer_df.to_parquet('data/customer_features.parquet', index=False)

    mo.md(f"""
    ✅ **数据已保存到 `data/customer_features.parquet`**

    - 文件大小: {os.path.getsize('data/customer_features.parquet') / 1024:.1f} KB
    - 记录数: {len(customer_df):,}
    - 列数: {len(customer_df.columns)}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 🎯 创建Feast配置

    现在我们需要创建Feast的配置文件。在实际项目中，这通常是一个单独的Python文件。
    """
    )
    return


@app.cell
def _(
    churn_prediction_service,
    customer,
    customer_behavior,
    customer_demographics,
):
    # 创建feature_store.py配置（模拟）
    feature_store_config = f"""
    # feature_store.py - Feast配置文件

    from feast import Entity, FeatureView, Field, FileSource, FeatureService
    from feast.types import Float32, Int64, String
    from datetime import timedelta

    # 定义实体
    {repr(customer)}

    # 定义数据源
    customer_source = FileSource(
    path="data/customer_features.parquet",
    timestamp_field="event_timestamp"
    )

    # 定义特征视图
    {repr(customer_demographics)}

    {repr(customer_behavior)}

    # 定义特征服务
    {repr(churn_prediction_service)}
    """

    print("📝 Feast配置文件内容:")
    print("=" * 50)
    print(feature_store_config[:500] + "...")
    print("=" * 50)

    # 保存配置文件
    with open('feature_store.py', 'w', encoding='utf-8') as f:
        f.write(feature_store_config)

    print("✅ 配置文件已保存为 feature_store.py")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 🚀 初始化Feast存储

    现在让我们初始化Feast特征存储并应用我们的配置。
    """
    )
    return


@app.cell
def _():
    # 初始化Feast存储
    try:
        from feast import FeatureStore

        # 创建特征存储实例
        store = FeatureStore(repo_path=".")

        print("✅ Feast特征存储初始化成功")
        print(f"存储路径: {store.repo_path}")

        # 尝试应用配置（在实际环境中需要运行 feast apply）
        print("\n📋 可用的特征视图:")
        try:
            feature_views = store.list_feature_views()
            for fv in feature_views:
                print(f"  - {fv.name}: {len(fv.schema)} 个特征")
        except Exception as e:
            print("  ⚠️ 需要先运行 'feast apply' 来应用配置")
            print(f"  错误: {str(e)}")

    except Exception as e:
        print(f"❌ Feast初始化失败: {str(e)}")
        print("请确保已安装Feast: pip install feast")
        store = None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 📊 生成训练数据

    让我们创建训练标签并生成时间点正确的训练数据集。
    """
    )
    return


@app.cell
def _(datetime, np, pd):
    # 创建训练标签数据
    def create_training_labels():
        """创建训练标签数据"""
        np.random.seed(42)

        # 选择一些客户和时间点进行预测
        customer_ids = [f"customer_{i:04d}" for i in range(0, 50, 5)]  # 每5个客户选1个
        prediction_times = [
            datetime(2023, 2, 1),
            datetime(2023, 3, 1),
            datetime(2023, 4, 1)
        ]

        entity_rows = []
        for customer_id in customer_ids:
            for pred_time in prediction_times:
                # 模拟流失标签（基于一些业务逻辑）
                # 这里使用简单的随机生成，实际中会基于真实业务规则
                churn_prob = np.random.random()
                churn_label = 1 if churn_prob > 0.8 else 0

                entity_rows.append({
                    "customer_id": customer_id,
                    "event_timestamp": pred_time,
                    "churn_label": churn_label
                })

        return pd.DataFrame(entity_rows)

    # 创建实体DataFrame
    entity_df = create_training_labels()

    print("✅ 训练标签数据创建完成")
    print(f"📊 标签数据形状: {entity_df.shape}")
    print(f"👥 涉及客户数: {entity_df['customer_id'].nunique()}")
    print(f"📅 预测时间点: {entity_df['event_timestamp'].nunique()}")
    print(f"⚖️ 流失率: {entity_df['churn_label'].mean():.1%}")

    print("\n📋 标签数据样本:")
    entity_df.head(10)
    return (entity_df,)


@app.cell
def _(customer_df, entity_df, pd):
    # 模拟历史特征获取（因为可能没有完整的Feast环境）
    def simulate_historical_features(entity_df, customer_df):
        """模拟历史特征获取过程"""

        print("🔄 模拟历史特征获取...")

        # 为每个实体行找到对应的历史特征
        training_rows = []

        for _, row in entity_df.iterrows():
            customer_id = row['customer_id']
            event_timestamp = row['event_timestamp']
            churn_label = row['churn_label']

            # 找到该客户在该时间点之前的最新特征
            customer_features = customer_df[
                (customer_df['customer_id'] == customer_id) &
                (customer_df['event_timestamp'] <= event_timestamp)
            ].sort_values('event_timestamp').tail(1)

            if not customer_features.empty:
                feature_row = customer_features.iloc[0].to_dict()
                feature_row['churn_label'] = churn_label
                training_rows.append(feature_row)

        return pd.DataFrame(training_rows)

    # 生成训练数据
    training_df = simulate_historical_features(entity_df, customer_df)

    print("✅ 训练数据生成完成")
    print(f"📊 训练数据形状: {training_df.shape}")
    print(f"🎯 特征列数: {len([col for col in training_df.columns if col != 'churn_label'])}")

    # 检查缺失值
    total_missing = training_df.isnull().sum().sum()
    if total_missing == 0:
        print("✅ 无缺失值")
    else:
        print(f"⚠️ 发现 {total_missing} 个缺失值")

    print("\n📋 训练数据样本:")
    training_df.head()
    return total_missing, training_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 🤖 模型训练

    现在让我们使用生成的训练数据来训练一个客户流失预测模型。
    """
    )
    return


@app.cell
def _(pd, training_df):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
    import joblib

    def train_churn_model(training_df):
        """训练客户流失预测模型"""

        print("🤖 开始训练客户流失预测模型...")

        # 准备特征和标签
        feature_columns = [
            'age', 'income', 'credit_score', 'monthly_charges',
            'total_charges', 'support_calls', 'contract_length'
        ]

        X = training_df[feature_columns].fillna(0)  # 简单处理缺失值
        y = training_df['churn_label']

        print(f"📊 特征矩阵形状: {X.shape}")
        print(f"🎯 标签分布: 流失={y.sum()}, 未流失={len(y)-y.sum()}")

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 训练模型
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'  # 处理类别不平衡
        )

        model.fit(X_train_scaled, y_train)

        # 评估模型
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        print("\n📈 模型评估结果:")
        print("=" * 40)
        print(classification_report(y_test, y_pred))

        try:
            auc_score = roc_auc_score(y_test, y_pred_proba)
            print(f"🎯 AUC Score: {auc_score:.3f}")
        except ValueError:
            print("⚠️ 无法计算AUC（可能只有一个类别）")

        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n🔍 特征重要性:")
        for _, row in feature_importance.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")

        # 保存模型和预处理器
        joblib.dump(model, 'churn_model.pkl')
        joblib.dump(scaler, 'feature_scaler.pkl')

        print("\n✅ 模型训练完成并已保存")

        return model, scaler, feature_importance

    # 训练模型
    model, scaler, feature_importance = train_churn_model(training_df)
    return feature_importance, model, scaler


@app.cell
def _(feature_importance, mo):
    # 可视化特征重要性
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('重要性分数')
    plt.title('特征重要性排序')
    plt.gca().invert_yaxis()

    # 在marimo中显示图表
    mo.md(f"""
    ## 📊 特征重要性分析

    根据随机森林模型的分析，各特征的重要性排序如下：

    {feature_importance.to_string(index=False)}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 🌐 在线特征服务

    现在让我们模拟在线特征服务，展示如何在生产环境中进行实时预测。
    """
    )
    return


@app.cell
def _(customer_df, datetime, model, pd, scaler):
    def simulate_online_features(customer_ids, current_time=None):
        """模拟在线特征获取"""

        if current_time is None:
            current_time = datetime.now()

        print(f"🔄 模拟在线特征获取 (时间: {current_time})")

        # 模拟从在线存储获取最新特征
        online_features_list = []

        for customer_id in customer_ids:
            # 获取该客户的最新特征
            customer_features = customer_df[
                customer_df['customer_id'] == customer_id
            ].sort_values('event_timestamp').tail(1)

            if not customer_features.empty:
                features = customer_features.iloc[0].to_dict()
                features['customer_id'] = customer_id
                online_features_list.append(features)
            else:
                print(f"⚠️ 客户 {customer_id} 未找到特征数据")

        return pd.DataFrame(online_features_list)

    def predict_churn_online(customer_ids):
        """使用在线特征进行实时预测"""

        print(f"🎯 为 {len(customer_ids)} 个客户进行流失预测...")

        # 获取在线特征
        features_df = simulate_online_features(customer_ids)

        if features_df.empty:
            print("❌ 未获取到特征数据")
            return []

        print(f"✅ 获取到 {len(features_df)} 个客户的特征")

        # 准备预测数据
        feature_columns = [
            'age', 'income', 'credit_score', 'monthly_charges',
            'total_charges', 'support_calls', 'contract_length'
        ]

        X = features_df[feature_columns].fillna(0)

        # 加载模型和预处理器
        try:
            # 使用已训练的模型
            X_scaled = scaler.transform(X)
            predictions = model.predict_proba(X_scaled)[:, 1]

            # 生成结果
            results = []
            for i, customer_id in enumerate(features_df['customer_id']):
                risk_level = (
                    'High' if predictions[i] > 0.7
                    else 'Medium' if predictions[i] > 0.3
                    else 'Low'
                )

                results.append({
                    'customer_id': customer_id,
                    'churn_probability': predictions[i],
                    'risk_level': risk_level,
                    'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })

            return results

        except Exception as e:
            print(f"❌ 预测失败: {str(e)}")
            return []

    # 进行在线预测演示
    test_customers = ["customer_0000", "customer_0005", "customer_0010", "customer_0015"]
    predictions = predict_churn_online(test_customers)

    print("\n🎯 实时预测结果:")
    print("=" * 60)
    for pred in predictions:
        print(f"客户 {pred['customer_id']}: "
              f"流失概率 {pred['churn_probability']:.3f} "
              f"({pred['risk_level']} Risk)")
    print("=" * 60)

    # 转换为DataFrame便于显示
    predictions_df = pd.DataFrame(predictions)
    predictions_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 🔍 特征监控和质量检查

    在生产环境中，监控特征质量是非常重要的。让我们实现一些基本的监控功能。
    """
    )
    return


@app.cell
def _(customer_df, timedelta):
    def monitor_feature_quality(df, time_window_days=7):
        """监控特征质量"""

        print(f"🔍 特征质量监控 (最近 {time_window_days} 天)")
        print("=" * 50)

        # 计算时间窗口
        end_date = df['event_timestamp'].max()
        start_date = end_date - timedelta(days=time_window_days)

        recent_data = df[df['event_timestamp'] >= start_date]

        print(f"📅 监控时间范围: {start_date.date()} 到 {end_date.date()}")
        print(f"📊 监控数据量: {len(recent_data)} 条记录")

        # 数据质量检查
        quality_report = {}

        numeric_columns = ['age', 'income', 'credit_score', 'monthly_charges', 'total_charges', 'support_calls']

        for col in numeric_columns:
            col_data = recent_data[col]

            quality_report[col] = {
                'missing_rate': col_data.isnull().mean(),
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'outlier_rate': ((col_data < col_data.quantile(0.01)) |
                               (col_data > col_data.quantile(0.99))).mean()
            }

        # 显示质量报告
        print("\n📋 特征质量报告:")
        for feature, stats in quality_report.items():
            print(f"\n🔧 {feature}:")
            print(f"  缺失率: {stats['missing_rate']:.1%}")
            print(f"  均值: {stats['mean']:.2f}")
            print(f"  标准差: {stats['std']:.2f}")
            print(f"  异常值率: {stats['outlier_rate']:.1%}")

            # 质量警告
            if stats['missing_rate'] > 0.1:
                print(f"  ⚠️ 警告: 缺失率过高 ({stats['missing_rate']:.1%})")
            if stats['outlier_rate'] > 0.05:
                print(f"  ⚠️ 警告: 异常值过多 ({stats['outlier_rate']:.1%})")

        return quality_report

    # 执行特征质量监控
    quality_report = monitor_feature_quality(customer_df)
    return


@app.cell
def _(customer_df, pd):
    def detect_data_drift(df, reference_period_days=30, current_period_days=7):
        """检测数据漂移"""

        print("🌊 数据漂移检测")
        print("=" * 40)

        end_date = df['event_timestamp'].max()

        # 参考期间（基线）
        ref_start = end_date - pd.Timedelta(days=reference_period_days + current_period_days)
        ref_end = end_date - pd.Timedelta(days=current_period_days)
        reference_data = df[(df['event_timestamp'] >= ref_start) & (df['event_timestamp'] < ref_end)]

        # 当前期间
        current_start = end_date - pd.Timedelta(days=current_period_days)
        current_data = df[df['event_timestamp'] >= current_start]

        print(f"📊 参考期间: {ref_start.date()} 到 {ref_end.date()} ({len(reference_data)} 条记录)")
        print(f"📊 当前期间: {current_start.date()} 到 {end_date.date()} ({len(current_data)} 条记录)")

        # 比较统计特征
        numeric_columns = ['age', 'income', 'credit_score', 'monthly_charges', 'total_charges']

        drift_report = {}

        for col in numeric_columns:
            ref_mean = reference_data[col].mean()
            current_mean = current_data[col].mean()

            ref_std = reference_data[col].std()
            current_std = current_data[col].std()

            # 计算变化百分比
            mean_change = (current_mean - ref_mean) / ref_mean * 100 if ref_mean != 0 else 0
            std_change = (current_std - ref_std) / ref_std * 100 if ref_std != 0 else 0

            drift_report[col] = {
                'mean_change_pct': mean_change,
                'std_change_pct': std_change,
                'drift_detected': abs(mean_change) > 10 or abs(std_change) > 20
            }

        print("\n📈 漂移检测结果:")
        for feature, stats in drift_report.items():
            status = "🚨 检测到漂移" if stats['drift_detected'] else "✅ 正常"
            print(f"\n{feature}: {status}")
            print(f"  均值变化: {stats['mean_change_pct']:+.1f}%")
            print(f"  标准差变化: {stats['std_change_pct']:+.1f}%")

        return drift_report

    # 执行数据漂移检测
    drift_report = detect_data_drift(customer_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 📚 Feast最佳实践

    基于我们的实际案例，让我们总结一些Feast使用的最佳实践。
    """
    )
    return


@app.cell
def _(mo):
    best_practices = {
        "特征命名规范": [
            "✅ 使用描述性名称：customer_age_years 而不是 age",
            "✅ 包含时间窗口：purchases_30d, clicks_7d",
            "✅ 使用一致的命名约定：snake_case",
            "✅ 避免缩写：monthly_revenue 而不是 mon_rev"
        ],

        "特征组织": [
            "✅ 按业务域分组：customer_features, product_features",
            "✅ 按更新频率分组：daily_features, realtime_features",
            "✅ 按数据源分组：database_features, api_features",
            "✅ 使用标签进行分类和搜索"
        ],

        "数据质量": [
            "✅ 实施特征验证：数据类型、范围检查",
            "✅ 监控特征分布变化：数据漂移检测",
            "✅ 设置数据质量警报：缺失值、异常值",
            "✅ 定期审查特征使用情况"
        ],

        "性能优化": [
            "✅ 选择合适的TTL：平衡新鲜度和存储成本",
            "✅ 优化批处理窗口：减少计算开销",
            "✅ 使用适当的分区策略：提高查询性能",
            "✅ 监控存储和计算成本"
        ],

        "安全和治理": [
            "✅ 实施访问控制：基于角色的特征访问",
            "✅ 数据血缘跟踪：了解特征来源和依赖",
            "✅ 合规性检查：确保符合数据保护法规",
            "✅ 审计日志：跟踪特征使用和修改"
        ]
    }

    practices_md = "## 🎯 Feast最佳实践总结\n\n"

    for category, items in best_practices.items():
        practices_md += f"### 📋 {category}\n\n"
        for item in items:
            practices_md += f"- {item}\n"
        practices_md += "\n"

    mo.md(practices_md)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🎪 总结

    通过这个交互式指南，我们深入探索了Feast特征存储的核心概念和实际应用：

    ### 🎯 **关键学习成果**

    1. **理解了Feast架构**：Entity、Feature View、Data Source、Feature Service的作用和关系
    2. **掌握了完整工作流**：从特征定义到模型部署的端到端流程
    3. **实现了实际案例**：客户流失预测的完整实现
    4. **学习了监控技术**：特征质量监控和数据漂移检测
    5. **掌握了最佳实践**：特征命名、组织、质量保证等

    ### 🚀 **Feast的核心价值**

    - **一致性保证**：训练和服务使用相同的特征定义
    - **时间正确性**：防止数据泄漏，确保历史数据的准确性
    - **开发效率**：特征重用，避免重复开发
    - **运维简化**：统一的特征管理和监控
    - **可扩展性**：支持从原型到生产的无缝扩展

    ### 🔮 **下一步行动**

    1. **安装Feast**：`pip install feast`
    2. **创建项目**：`feast init your_project`
    3. **定义特征**：根据业务需求设计特征视图
    4. **应用配置**：`feast apply`
    5. **开始使用**：在训练和服务中集成Feast

    Feast是现代MLOps架构中不可或缺的组件，它将特征管理从临时性的工作转变为系统性的能力，为构建可靠、可扩展的ML系统奠定了坚实基础！🎉
    """
    )
    return


if __name__ == "__main__":
    app.run()
