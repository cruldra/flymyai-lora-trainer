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
        r"""
    # 什么是特征？机器学习中的核心概念详解

    ## 🤔 特征到底是什么？

    **特征（Feature）** 是机器学习中用来描述数据的**可测量属性**或**特性**。简单来说，特征就是我们用来描述一个对象的各种信息。

    ### 🏠 生活中的例子

    想象你要**买房子**，你会关注哪些信息？

    - 🏡 **面积**：120平方米
    - 📍 **位置**：市中心
    - 🛏️ **房间数**：3室2厅
    - 🏢 **楼层**：8楼
    - 💰 **价格**：300万
    - 🚇 **距离地铁**：500米
    - 🏫 **学区**：重点小学
    - 🎂 **房龄**：5年

    这些信息就是房子的**特征**！每个特征都描述了房子的一个方面。

    ### 🤖 在机器学习中

    机器学习模型就像一个"专家"，它通过学习大量的特征数据，来做出预测或决策。

    比如：
    - 📧 **垃圾邮件检测**：邮件长度、关键词数量、发件人信息
    - 🛒 **商品推荐**：用户年龄、购买历史、浏览记录
    - 🏥 **疾病诊断**：体温、血压、症状描述
    - 📈 **股票预测**：历史价格、交易量、市场指标
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📊 特征的类型

    特征可以分为不同的类型，让我们通过实际例子来理解：
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import pandas as pd
    import numpy as np

    # 创建一个示例数据集：学生信息
    np.random.seed(42)

    students_data = {
        # 数值特征
        'age': np.random.randint(18, 25, 100),  # 年龄
        'height': np.random.normal(170, 10, 100),  # 身高(cm)
        'weight': np.random.normal(65, 15, 100),  # 体重(kg)
        'study_hours': np.random.normal(6, 2, 100),  # 每日学习时间
        'exam_score': np.random.normal(75, 15, 100),  # 考试成绩

        # 类别特征
        'gender': np.random.choice(['男', '女'], 100),  # 性别
        'major': np.random.choice(['计算机', '数学', '物理', '化学'], 100),  # 专业
        'city': np.random.choice(['北京', '上海', '广州', '深圳'], 100),  # 城市

        # 布尔特征
        'has_scholarship': np.random.choice([True, False], 100),  # 是否有奖学金
        'is_international': np.random.choice([True, False], 100, p=[0.1, 0.9]),  # 是否国际学生
    }

    # 创建DataFrame
    df = pd.DataFrame(students_data)

    # 确保数据合理性
    df['height'] = df['height'].clip(150, 200)
    df['weight'] = df['weight'].clip(40, 100)
    df['study_hours'] = df['study_hours'].clip(1, 12)
    df['exam_score'] = df['exam_score'].clip(0, 100)

    print("📚 学生数据集示例")
    print(f"数据形状: {df.shape}")
    print("\n前5行数据:")
    df.head()
    return df, pd, students_data


@app.cell(hide_code=True)
def _(df, mo):
    mo.md(
        f"""
    ### 🔢 数值特征（Numerical Features）

    数值特征是可以用数字表示的特征，可以进行数学运算。

    **连续数值特征**：
    - 身高: {df['height'].min():.1f}cm - {df['height'].max():.1f}cm
    - 体重: {df['weight'].min():.1f}kg - {df['weight'].max():.1f}kg
    - 学习时间: {df['study_hours'].min():.1f}小时 - {df['study_hours'].max():.1f}小时

    **离散数值特征**：
    - 年龄: {df['age'].min()}岁 - {df['age'].max()}岁
    - 考试成绩: {df['exam_score'].min():.0f}分 - {df['exam_score'].max():.0f}分
    """
    )
    return


@app.cell(hide_code=True)
def _(df):
    # 数值特征的统计分析
    print("📊 数值特征统计信息:")
    numeric_features = ['age', 'height', 'weight', 'study_hours', 'exam_score']

    for feat_name in numeric_features:
        data = df[feat_name]
        print(f"\n🔢 {feat_name}:")
        print(f"  平均值: {data.mean():.2f}")
        print(f"  标准差: {data.std():.2f}")
        print(f"  最小值: {data.min():.2f}")
        print(f"  最大值: {data.max():.2f}")
        print(f"  中位数: {data.median():.2f}")

    # 显示数值特征的分布
    df[numeric_features].describe()
    return numeric_features,


@app.cell(hide_code=True)
def _(df, mo):
    mo.md(
        f"""
    ### 🏷️ 类别特征（Categorical Features）

    类别特征表示不同的类别或组别，不能直接进行数学运算。

    **性别分布**：
    - 男: {(df['gender'] == '男').sum()}人 ({(df['gender'] == '男').mean()*100:.1f}%)
    - 女: {(df['gender'] == '女').sum()}人 ({(df['gender'] == '女').mean()*100:.1f}%)

    **专业分布**：
    {df['major'].value_counts().to_string()}

    **城市分布**：
    {df['city'].value_counts().to_string()}
    """
    )
    return


@app.cell(hide_code=True)
def _(df):
    # 类别特征的分析
    print("🏷️ 类别特征分析:")
    categorical_features = ['gender', 'major', 'city']

    for cat_feat in categorical_features:
        print(f"\n📋 {cat_feat}:")
        value_counts = df[cat_feat].value_counts()
        for cat_value, count in value_counts.items():
            percentage = count / len(df) * 100
            print(f"  {cat_value}: {count}人 ({percentage:.1f}%)")
        print(f"  唯一值数量: {df[cat_feat].nunique()}")

    return categorical_features,


@app.cell(hide_code=True)
def _(df, mo):
    mo.md(
        f"""
    ### ✅ 布尔特征（Boolean Features）

    布尔特征只有两个值：True/False 或 是/否。

    **奖学金情况**：
    - 有奖学金: {df['has_scholarship'].sum()}人 ({df['has_scholarship'].mean()*100:.1f}%)
    - 无奖学金: {(~df['has_scholarship']).sum()}人 ({(~df['has_scholarship']).mean()*100:.1f}%)

    **国际学生情况**：
    - 国际学生: {df['is_international'].sum()}人 ({df['is_international'].mean()*100:.1f}%)
    - 本地学生: {(~df['is_international']).sum()}人 ({(~df['is_international']).mean()*100:.1f}%)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🔧 特征工程：让特征更有用

    **特征工程**是从原始数据中创建新特征的过程，这是机器学习中最重要的技能之一！

    ### 为什么需要特征工程？

    原始数据往往不能直接用于机器学习，我们需要：
    - 🔄 **转换数据格式**：让机器能理解
    - 📈 **提取有用信息**：发现隐藏的模式
    - 🎯 **提高预测能力**：让模型更准确
    """
    )
    return


@app.cell(hide_code=True)
def _(df, pd, students_data):
    # 特征工程示例
    print("🔧 特征工程示例:")

    # 1. 创建新的数值特征
    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2  # BMI指数
    df['study_efficiency'] = df['exam_score'] / df['study_hours']  # 学习效率

    # 2. 创建类别特征
    df['age_group'] = pd.cut(df['age'], bins=[17, 20, 22, 25], labels=['年轻', '中等', '成熟'])
    df['score_level'] = pd.cut(df['exam_score'], bins=[0, 60, 80, 100], labels=['不及格', '良好', '优秀'])

    # 3. 创建布尔特征
    df['is_tall'] = df['height'] > df['height'].median()  # 是否高个子
    df['studies_a_lot'] = df['study_hours'] > 8  # 是否学习时间长
    df['high_achiever'] = (df['exam_score'] > 85) & df['has_scholarship']  # 是否高成就者

    print("✅ 新特征创建完成!")
    print(f"原始特征数: {len(students_data)}")
    print(f"现在特征数: {len(df.columns)}")
    print(f"新增特征数: {len(df.columns) - len(students_data)}")

    print("\n🆕 新创建的特征:")
    new_features = ['bmi', 'study_efficiency', 'age_group', 'score_level', 'is_tall', 'studies_a_lot', 'high_achiever']
    for new_feat in new_features:
        if df[new_feat].dtype in ['float64', 'int64']:
            print(f"  {new_feat}: {df[new_feat].mean():.2f} (平均值)")
        else:
            print(f"  {new_feat}: {df[new_feat].value_counts().to_dict()}")

    return new_features,


@app.cell(hide_code=True)
def _(df):
    # 显示特征工程后的数据样本
    print("📊 特征工程后的数据样本:")
    selected_columns = ['age', 'height', 'weight', 'exam_score', 'bmi', 'study_efficiency', 'age_group', 'score_level', 'high_achiever']
    df[selected_columns].head(10)
    return selected_columns,


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🎯 特征在机器学习中的作用

    让我们通过一个实际的预测任务来理解特征的重要性：**预测学生是否能获得奖学金**
    """
    )
    return


@app.cell(hide_code=True)
def _(df):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report

    # 准备数据进行机器学习
    print("🤖 准备机器学习数据...")

    # 选择特征
    feature_columns = [
        'age', 'height', 'weight', 'study_hours', 'exam_score',  # 原始数值特征
        'bmi', 'study_efficiency',  # 工程特征
        'gender', 'major', 'city',  # 类别特征
        'is_international', 'is_tall', 'studies_a_lot'  # 布尔特征
    ]

    # 目标变量
    target = 'has_scholarship'

    # 处理类别特征（编码）
    df_ml = df.copy()
    label_encoders = {}

    for col in ['gender', 'major', 'city']:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col])
        label_encoders[col] = le

    # 准备特征矩阵和目标向量
    X = df_ml[feature_columns]
    y = df_ml[target]

    print(f"✅ 特征矩阵形状: {X.shape}")
    print(f"✅ 目标向量形状: {y.shape}")
    print(f"✅ 特征列表: {feature_columns}")

    return (
        LabelEncoder,
        RandomForestClassifier,
        X,
        accuracy_score,
        classification_report,
        df_ml,
        feature_columns,
        label_encoders,
        target,
        train_test_split,
        y,
    )


@app.cell(hide_code=True)
def _(
    RandomForestClassifier,
    X,
    accuracy_score,
    classification_report,
    train_test_split,
    y,
):
    # 训练机器学习模型
    print("🚀 训练机器学习模型...")

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 评估
    accuracy = accuracy_score(y_test, y_pred)

    print(f"✅ 模型训练完成!")
    print(f"📊 准确率: {accuracy:.3f}")
    print(f"📈 训练集大小: {len(X_train)}")
    print(f"📈 测试集大小: {len(X_test)}")

    print("\n📋 详细评估报告:")
    print(classification_report(y_test, y_pred))

    return X_test, X_train, accuracy, model, y_pred, y_test, y_train


@app.cell(hide_code=True)
def _(feature_columns, model, pd):
    # 特征重要性分析
    print("🔍 特征重要性分析:")
    print("=" * 50)

    # 获取特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("📊 特征重要性排序:")
    for rank_num, (_, row) in enumerate(feature_importance.iterrows(), 1):
        print(f"{rank_num:2d}. {row['feature']:20s}: {row['importance']:.4f}")

    print(f"\n🎯 最重要的3个特征:")
    top_features = feature_importance.head(3)
    for _, row in top_features.iterrows():
        print(f"   • {row['feature']}: {row['importance']:.4f}")

    return feature_importance, top_features


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 💡 特征的重要性总结

    通过上面的例子，我们可以看到：

    ### 🎯 **什么是好特征？**

    1. **相关性强**：与预测目标密切相关
    2. **信息丰富**：能提供独特的信息
    3. **稳定可靠**：不会随时间剧烈变化
    4. **易于获取**：在实际应用中容易收集

    ### 🔧 **特征工程的价值**

    - **BMI指数**：比单独的身高体重更有意义
    - **学习效率**：结合了成绩和时间两个维度
    - **年龄分组**：将连续变量转为更有意义的类别
    - **组合特征**：如"高成就者"结合了多个条件

    ### 📈 **特征对模型的影响**

    - 好的特征可以显著提高模型准确性
    - 特征工程往往比选择复杂算法更重要
    - 领域知识在特征设计中至关重要
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🚀 实际应用中的特征例子

    让我们看看不同领域中常用的特征：
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # 不同领域的特征例子
    domain_features = {
        "🛒 电商推荐": [
            "用户年龄、性别、地理位置",
            "购买历史、浏览记录、收藏商品",
            "价格敏感度、品牌偏好",
            "购买时间模式、设备类型",
            "评价行为、社交网络信息"
        ],

        "🏥 医疗诊断": [
            "患者基本信息：年龄、性别、BMI",
            "生理指标：血压、心率、体温",
            "实验室检查：血常规、生化指标",
            "症状描述：疼痛程度、持续时间",
            "病史信息：家族史、既往病史"
        ],

        "💰 金融风控": [
            "个人信息：年龄、收入、职业",
            "信用历史：还款记录、逾期次数",
            "资产状况：存款、房产、投资",
            "行为特征：交易频率、消费模式",
            "社交网络：联系人信用状况"
        ],

        "🚗 自动驾驶": [
            "传感器数据：摄像头、雷达、激光",
            "车辆状态：速度、方向、油量",
            "环境信息：天气、路况、交通",
            "地图数据：道路类型、限速标志",
            "历史轨迹：常用路线、驾驶习惯"
        ],

        "📱 社交媒体": [
            "用户画像：年龄、兴趣、活跃度",
            "内容特征：文本、图片、视频",
            "互动行为：点赞、评论、分享",
            "网络关系：好友数量、影响力",
            "时间模式：发布时间、在线时长"
        ]
    }

    features_md = "### 🌍 不同领域的特征应用\n\n"

    for domain, feat_list in domain_features.items():
        features_md += f"#### {domain}\n\n"
        for feat_item in feat_list:
            features_md += f"- **{feat_item}**\n"
        features_md += "\n"

    mo.md(features_md)
    return domain_features,


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🎯 特征选择：如何选择最好的特征？

    并不是特征越多越好！我们需要选择最有用的特征。
    """
    )
    return


@app.cell(hide_code=True)
def _(X, feature_columns, y):
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.feature_selection import RFE

    print("🔍 特征选择方法演示:")
    print("=" * 40)

    # 方法1: 统计检验选择特征
    print("📊 方法1: 统计检验特征选择")
    selector_stats = SelectKBest(score_func=f_classif, k=5)
    X_selected_stats = selector_stats.fit_transform(X, y)

    # 获取选中的特征
    selected_features_stats = [feature_columns[i] for i in selector_stats.get_support(indices=True)]
    feature_scores = selector_stats.scores_

    print(f"选中的特征: {selected_features_stats}")
    print("特征得分:")
    for feat_idx, (sel_feat, score) in enumerate(zip(feature_columns, feature_scores)):
        status_icon = "✅" if sel_feat in selected_features_stats else "❌"
        print(f"  {status_icon} {sel_feat:20s}: {score:.2f}")
    return (
        RFE,
        SelectKBest,
        X_selected_stats,
        f_classif,
        feature_scores,
        selected_features_stats,
        selector_stats,
    )


@app.cell(hide_code=True)
def _(RFE, RandomForestClassifier, X, feature_columns, y):
    # 方法2: 递归特征消除
    print("\n🔄 方法2: 递归特征消除 (RFE)")

    # 使用随机森林作为基础估计器
    estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    selector_rfe = RFE(estimator=estimator, n_features_to_select=5)
    X_selected_rfe = selector_rfe.fit_transform(X, y)

    # 获取选中的特征
    selected_features_rfe = [feature_columns[i] for i in selector_rfe.get_support(indices=True)]
    feature_rankings = selector_rfe.ranking_

    print(f"选中的特征: {selected_features_rfe}")
    print("特征排名 (1=最重要):")
    for feat_col, rank in zip(feature_columns, feature_rankings):
        rank_status = "✅" if rank == 1 else f"#{rank}"
        print(f"  {rank_status} {feat_col:20s}: 排名 {rank}")
    return (
        X_selected_rfe,
        estimator,
        feature_rankings,
        selected_features_rfe,
        selector_rfe,
    )


@app.cell(hide_code=True)
def _(selected_features_rfe, selected_features_stats):
    # 比较不同方法的结果
    print("\n🔍 特征选择方法对比:")
    print("=" * 50)

    print("统计检验选择的特征:")
    for feat_stat in selected_features_stats:
        print(f"  ✅ {feat_stat}")

    print("\nRFE选择的特征:")
    for feat_rfe in selected_features_rfe:
        print(f"  ✅ {feat_rfe}")

    # 找出共同选择的特征
    common_features = set(selected_features_stats) & set(selected_features_rfe)
    print(f"\n🎯 两种方法都选择的特征:")
    for feat_common in common_features:
        print(f"  ⭐ {feat_common}")

    print(f"\n📊 总结:")
    print(f"  统计检验选择: {len(selected_features_stats)} 个特征")
    print(f"  RFE选择: {len(selected_features_rfe)} 个特征")
    print(f"  共同选择: {len(common_features)} 个特征")
    return common_features,


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ⚠️ 特征使用中的常见问题

    在实际应用中，我们需要避免一些常见的陷阱：
    """
    )
    return


@app.cell(hide_code=True)
def _(df, mo):
    # 演示常见问题
    problems_examples = {
        "🕳️ 数据泄漏": {
            "问题": "使用了未来信息或目标变量相关的信息",
            "错误示例": "用'是否获得奖学金'来预测'是否获得奖学金'",
            "正确做法": "只使用预测时刻之前的信息"
        },

        "🔢 多重共线性": {
            "问题": "特征之间高度相关，提供重复信息",
            "错误示例": "同时使用身高(cm)和身高(m)作为特征",
            "正确做法": "检测并移除高度相关的特征"
        },

        "📊 数据不平衡": {
            "问题": "某些类别的样本数量过少",
            "错误示例": f"奖学金获得者只有{df['has_scholarship'].sum()}人，占{df['has_scholarship'].mean()*100:.1f}%",
            "正确做法": "使用重采样或调整类别权重"
        },

        "🎯 过拟合": {
            "问题": "特征过多，模型记住了训练数据的噪声",
            "错误示例": "使用100个特征预测100个样本",
            "正确做法": "特征选择、正则化、交叉验证"
        },

        "📏 量纲不统一": {
            "问题": "不同特征的数值范围差异巨大",
            "错误示例": f"年龄范围{df['age'].min()}-{df['age'].max()}，收入可能是0-100000",
            "正确做法": "标准化或归一化特征"
        }
    }

    problems_md = "### ⚠️ 常见问题和解决方案\n\n"

    for problem_type, details in problems_examples.items():
        problems_md += f"#### {problem_type}\n\n"
        problems_md += f"**问题描述**: {details['问题']}\n\n"
        problems_md += f"**❌ 错误示例**: {details['错误示例']}\n\n"
        problems_md += f"**✅ 正确做法**: {details['正确做法']}\n\n"

    mo.md(problems_md)
    return problems_examples,


@app.cell(hide_code=True)
def _(df):
    # 演示特征相关性检查
    print("🔍 特征相关性分析:")
    print("=" * 40)

    # 计算数值特征之间的相关性
    numeric_cols = ['age', 'height', 'weight', 'study_hours', 'exam_score', 'bmi', 'study_efficiency']
    correlation_matrix = df[numeric_cols].corr()

    print("📊 高相关性特征对 (|相关系数| > 0.7):")

    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:
                feat1 = correlation_matrix.columns[i]
                feat2 = correlation_matrix.columns[j]
                high_corr_pairs.append((feat1, feat2, corr_value))
                print(f"  {feat1} ↔ {feat2}: {corr_value:.3f}")

    if not high_corr_pairs:
        print("  ✅ 没有发现高相关性特征对")

    print(f"\n📈 相关性矩阵 (部分):")
    print(correlation_matrix.round(3))
    return correlation_matrix, corr_value, high_corr_pairs, numeric_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🎓 特征工程的实用技巧

    基于实际经验，这里是一些特征工程的实用建议：
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # 实用技巧总结
    practical_tips = {
        "🎯 特征创建技巧": [
            "**组合特征**: 将多个特征组合，如 BMI = 体重/(身高²)",
            "**比率特征**: 创建比率，如 学习效率 = 成绩/学习时间",
            "**分箱特征**: 将连续变量分组，如 年龄组 = [年轻,中等,成熟]",
            "**时间特征**: 提取时间信息，如 星期几、月份、季节",
            "**统计特征**: 计算历史统计，如 平均值、最大值、趋势"
        ],

        "🔍 特征选择策略": [
            "**业务理解**: 基于领域知识选择相关特征",
            "**统计方法**: 使用相关性、卡方检验等统计方法",
            "**模型方法**: 使用特征重要性、递归消除等",
            "**实验验证**: 通过A/B测试验证特征效果",
            "**简单优先**: 优先选择简单、稳定的特征"
        ],

        "⚡ 性能优化": [
            "**特征缓存**: 缓存计算复杂的特征",
            "**增量计算**: 只计算新增数据的特征",
            "**并行处理**: 并行计算独立的特征",
            "**数据类型**: 使用合适的数据类型节省内存",
            "**特征存储**: 使用特征存储系统管理特征"
        ],

        "🛡️ 质量保证": [
            "**数据验证**: 检查特征的数据质量和完整性",
            "**异常检测**: 识别和处理异常值",
            "**一致性检查**: 确保训练和预测时特征一致",
            "**监控告警**: 监控特征分布的变化",
            "**版本管理**: 管理特征定义的版本变化"
        ]
    }

    tips_md = "### 💡 实用技巧总结\n\n"

    for tip_category, tips_list in practical_tips.items():
        tips_md += f"#### {tip_category}\n\n"
        for tip_item in tips_list:
            tips_md += f"- {tip_item}\n"
        tips_md += "\n"

    mo.md(tips_md)
    return practical_tips,


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🎉 总结：特征是机器学习的基石

    通过这个交互式指南，我们深入理解了特征的概念和重要性：

    ### 🎯 **核心要点回顾**

    1. **特征定义**: 特征是用来描述数据对象的可测量属性
    2. **特征类型**: 数值特征、类别特征、布尔特征各有特点
    3. **特征工程**: 从原始数据创建有用特征是关键技能
    4. **特征选择**: 选择最相关的特征，避免冗余和噪声
    5. **实际应用**: 不同领域有不同的特征设计策略

    ### 💡 **关键洞察**

    - **数据质量决定模型上限**: 再好的算法也无法弥补差的特征
    - **领域知识很重要**: 理解业务才能设计好特征
    - **简单往往更好**: 简单、稳定的特征通常更可靠
    - **持续优化**: 特征工程是一个迭代改进的过程

    ### 🚀 **下一步学习**

    现在你已经理解了特征的基本概念，可以继续学习：

    - **高级特征工程技术**: 深度特征交叉、自动特征生成
    - **特征存储系统**: 如Feast等工具的使用
    - **特征监控**: 生产环境中的特征质量监控
    - **AutoML**: 自动化的特征工程和选择

    记住：**好的特征是成功机器学习项目的一半！** 🎯
    """
    )
    return


if __name__ == "__main__":
    app.run()
