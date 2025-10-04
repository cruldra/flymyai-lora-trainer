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
    # 完整的MLOps蓝图：数据和管道工程—第B部分（含实现）

    MLOps和LLMOps速成课程—第6部分

    ## 回顾

    在这个MLOps和LLMOps速成课程的第5部分中，我们探索了所有机器学习系统基石的基础：数据和管道。

    ![数据管道基础](https://www.dailydoseofds.com/content/images/2025/08/image-143.png)

    我们首先介绍了机器学习中的数据景观，并探索了为什么数据及其处理在MLOps世界中如此重要。

    ![数据重要性](https://www.dailydoseofds.com/content/images/2025/08/image-144.png)

    然后我们继续探索生产ML系统中重要的不同类型数据源，如用户输入数据、系统日志、内部数据库和第三方源。

    ![数据源类型](https://www.dailydoseofds.com/content/images/2025/08/image-145.png)

    我们还讨论了数据源的各种分类，如文本vs二进制和行主序vs列主序格式。我们研究了它们的差异以及各自的优缺点。

    ![数据格式对比](https://www.dailydoseofds.com/content/images/2025/08/image-146.png)

    接下来，我们进入了第5部分的核心概念焦点，涵盖了ETL和ELT管道。我们探索了两种方法，并清楚地理解了它们的差异，同时仍然相互补充。

    ![ETL vs ELT](https://www.dailydoseofds.com/content/images/2025/08/image-147.png)

    从那里，我们进行了实际操作。我们详细演练了混合ETL/ELT管道的模拟，模拟了多个数据源以及提取、验证、转换和加载阶段。

    ![实际实现](https://www.dailydoseofds.com/content/images/2025/08/image-148.png)

    如果你还没有探索第5部分，我们强烈建议先阅读它，因为它奠定了概念框架和实现理解，这将帮助你更好地理解我们即将深入的内容。

    在本章中，我们将继续ML系统中的数据处理和管理，深入探讨概念和实际实现。

    我们将涵盖蓝图，学习如何专门为机器学习设计和采样数据，并深入探讨数据泄漏这一危险陷阱。然后我们将在特征存储中集中我们的工作，这是确保训练和服务之间一致性的中心。

    一如既往，每个概念都将得到具体示例、演练和实用技巧的支持，帮助你掌握想法和实现。

    让我们开始吧！

    ---

    ## 采样策略

    采样是从更大的数据池中选择数据子集的实践。在机器学习中，采样发生在工作流的许多阶段：

    - **选择要收集的真实世界数据**来构建你的数据集
    - **选择可用数据的子集**用于标注或训练（特别是当你拥有的数据超过可行使用量时）
    - **将数据分割**为训练/验证/测试集
    - **训练期间的数据采样**用于每个批次（例如，在随机梯度下降中）
    - **监控中的采样**（例如，只记录一部分预测用于分析）

    ![采样应用场景](https://www.dailydoseofds.com/content/images/2025/08/image-149.png)

    对大多数人来说，对采样的常见接触可能是训练/验证/测试分割。但重要的是要意识到，你如何采样可能会引入偏差并影响模型性能。

    ### 为什么采样很重要？

    在许多情况下，我们不能或不使用所有可用数据。也许数据太大（在万亿条记录上训练不可行），或者获得标签成本高昂，所以我们标注一个子集，或者我们故意下采样以进行更快的实验。

    显而易见的是，**好的采样可以使模型开发高效并确保模型泛化，而差的采样可能误导你的结果。**

    例如，选择一个不具代表性的子集可能导致你的模型在那个特定子集上表现良好，但在生产中失败。

    ### 采样类型

    广义上，采样方法分为两个家族：

    ![采样方法分类](https://www.dailydoseofds.com/content/images/2025/08/image-150.png)

    - **非概率采样**：不严格基于随机机会，而是使用一些主观或实际标准来选择数据
    - **概率采样**：总体中的每个数据点都有被选择的某种概率，通常努力获得无偏样本

    让我们看看每个类别中的常见技术及其含义：

    #### 非概率采样方法

    在非概率采样方法下，我们有：

    ##### 便利采样

    选择最容易获得的数据。例如，使用日志中的前10,000条记录，因为它们随手可得，或者使用从一个可访问源（如一个城市或一个用户组）收集的数据集，因为它很方便。

    ![便利采样](https://www.dailydoseofds.com/content/images/2025/08/image-152-1.png)
    *便利采样*

    **这种方法的含义**包括高偏差风险，因为样本可能不代表整体人群。

    这种方法很受欢迎，因为正如名称所说，它很方便，但它可能扭曲结果。例如，在单个城市数据上训练的模型可能无法泛化到其他地区。

    ##### 雪球采样

    使用现有样本数据来招募更多数据。这通常用于社交网络或图中。例如，你有一些用户的数据，然后你包括他们的朋友，然后朋友的朋友，等等。

    ![雪球采样](https://www.dailydoseofds.com/content/images/2025/08/image-155-1.png)
    *雪球采样*

    **优点**：对于难以接触的人群很有用（例如，稀有疾病患者）。

    **缺点**：可能创建高度相关的样本，因为网络中的人往往相似。这可能导致缺乏多样性和泛化问题。

    ##### 判断采样

    基于专家判断或特定标准选择数据。例如，医生可能选择他们认为"典型"的病例用于研究。

    **优点**：可以确保包含重要或代表性的案例。

    **缺点**：高度主观，可能引入专家偏见。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### 概率采样方法

    概率采样方法提供了更严格和统计上合理的数据选择方法：

    ##### 简单随机采样

    每个数据点都有相等的被选择概率。这是最基本的概率采样形式。

    ```python
    import pandas as pd
    import numpy as np

    # 简单随机采样示例
    def simple_random_sampling(data, sample_size):
        \"\"\"从数据中进行简单随机采样\"\"\"
        return data.sample(n=sample_size, random_state=42)

    # 示例使用
    # sampled_data = simple_random_sampling(full_dataset, 1000)
    ```

    **优点**：无偏，每个数据点机会相等。

    **缺点**：可能无法捕获重要的子组，特别是如果某些组在总体中很少见。

    ##### 分层采样

    将总体分为子组（层），然后从每层中采样。这确保所有重要子组都在样本中得到代表。

    ```python
    from sklearn.model_selection import train_test_split

    def stratified_sampling(data, target_column, sample_size, random_state=42):
        \"\"\"基于目标变量进行分层采样\"\"\"

        # 计算每个层的比例
        class_proportions = data[target_column].value_counts(normalize=True)

        sampled_data = []
        for class_value, proportion in class_proportions.items():
            class_data = data[data[target_column] == class_value]
            class_sample_size = int(sample_size * proportion)

            if class_sample_size > 0:
                class_sample = class_data.sample(
                    n=min(class_sample_size, len(class_data)), 
                    random_state=random_state
                )
                sampled_data.append(class_sample)

        return pd.concat(sampled_data, ignore_index=True)
    ```

    **优点**：确保重要子组的代表性，通常比简单随机采样更准确。

    **缺点**：需要事先了解相关的分层变量。

    ##### 系统采样

    选择每第k个数据点，其中k = 总体大小/样本大小。

    ```python
    def systematic_sampling(data, sample_size):
        \"\"\"系统采样实现\"\"\"
        n = len(data)
        k = n // sample_size  # 采样间隔

        # 随机选择起始点
        start = np.random.randint(0, k)

        # 选择每第k个元素
        indices = range(start, n, k)
        return data.iloc[list(indices)[:sample_size]]
    ```

    **优点**：简单实现，分布均匀。

    **缺点**：如果数据中存在周期性模式，可能引入偏差。

    ##### 聚类采样

    将总体分为聚类，然后随机选择整个聚类。

    ```python
    def cluster_sampling(data, cluster_column, num_clusters):
        \"\"\"聚类采样实现\"\"\"

        # 获取所有唯一聚类
        all_clusters = data[cluster_column].unique()

        # 随机选择聚类
        selected_clusters = np.random.choice(
            all_clusters, 
            size=min(num_clusters, len(all_clusters)), 
            replace=False
        )

        # 返回选定聚类中的所有数据
        return data[data[cluster_column].isin(selected_clusters)]
    ```

    **优点**：当聚类内部相似但聚类间不同时很有效，成本效益高。

    **缺点**：如果聚类内部差异很大，可能不够代表性。

    ### ML中的采样最佳实践

    #### 1. 时间序列数据的采样

    对于时间序列数据，**永远不要随机打乱**！使用时间基础的分割：

    ```python
    def temporal_split(data, time_column, train_ratio=0.7, val_ratio=0.15):
        \"\"\"时间序列数据的时间基础分割\"\"\"

        # 按时间排序
        data_sorted = data.sort_values(time_column)

        n = len(data_sorted)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_data = data_sorted.iloc[:train_end]
        val_data = data_sorted.iloc[train_end:val_end]
        test_data = data_sorted.iloc[val_end:]

        return train_data, val_data, test_data
    ```

    #### 2. 不平衡数据的采样

    对于类别不平衡的数据，考虑分层采样或专门的重采样技术：

    ```python
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    def handle_imbalanced_data(X, y, strategy='smote'):
        \"\"\"处理不平衡数据的采样策略\"\"\"

        if strategy == 'smote':
            sampler = SMOTE(random_state=42)
        elif strategy == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        else:
            raise ValueError("Strategy must be 'smote' or 'undersample'")

        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled
    ```

    #### 3. 大数据的采样

    对于大规模数据，考虑分层或分块采样：

    ```python
    def large_data_sampling(data, sample_size, chunk_size=10000):
        \"\"\"大数据的分块采样\"\"\"

        total_rows = len(data)
        sampling_ratio = sample_size / total_rows

        sampled_chunks = []

        # 分块处理
        for chunk_start in range(0, total_rows, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_rows)
            chunk = data.iloc[chunk_start:chunk_end]

            # 从每个块中采样
            chunk_sample_size = int(len(chunk) * sampling_ratio)
            if chunk_sample_size > 0:
                chunk_sample = chunk.sample(n=chunk_sample_size, random_state=42)
                sampled_chunks.append(chunk_sample)

        return pd.concat(sampled_chunks, ignore_index=True)
    ```

    ### 采样质量评估

    评估采样质量的关键指标：

    ```python
    def evaluate_sampling_quality(original_data, sampled_data, target_column):
        \"\"\"评估采样质量\"\"\"

        results = {}

        # 1. 类别分布比较
        orig_dist = original_data[target_column].value_counts(normalize=True)
        samp_dist = sampled_data[target_column].value_counts(normalize=True)

        results['distribution_difference'] = abs(orig_dist - samp_dist).mean()

        # 2. 统计特征比较
        numeric_columns = original_data.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            orig_mean = original_data[col].mean()
            samp_mean = sampled_data[col].mean()
            results[f'{col}_mean_diff'] = abs(orig_mean - samp_mean)

        # 3. 采样比例
        results['sampling_ratio'] = len(sampled_data) / len(original_data)

        return results
    ```

    采样是ML管道中的关键步骤，正确的采样策略可以显著影响模型的性能和泛化能力。选择合适的采样方法取决于你的数据特征、业务需求和计算资源。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 类别不平衡处理

    类别不平衡是机器学习中最常见的挑战之一，特别是在真实世界的应用中。当数据集中不同类别的样本数量差异很大时，就会出现类别不平衡问题。

    ### 什么是类别不平衡？

    类别不平衡指的是分类问题中各类别的样本数量分布不均匀的情况。例如：

    - **欺诈检测**：正常交易占99%，欺诈交易占1%
    - **医疗诊断**：健康患者占95%，患病患者占5%
    - **垃圾邮件检测**：正常邮件占90%，垃圾邮件占10%
    - **客户流失预测**：留存客户占85%，流失客户占15%

    ### 为什么类别不平衡是问题？

    #### 1. **准确率误导**
    ```python
    # 示例：99%正常，1%异常的数据
    # 一个总是预测"正常"的模型会有99%的准确率
    # 但它完全无法检测异常情况！

    def accuracy_paradox_example():
        \"\"\"展示准确率悖论\"\"\"

        # 模拟不平衡数据
        total_samples = 10000
        positive_samples = 100  # 1%
        negative_samples = 9900  # 99%

        # "愚蠢"分类器：总是预测多数类
        always_negative_predictions = [0] * total_samples
        true_labels = [1] * positive_samples + [0] * negative_samples

        # 计算准确率
        correct_predictions = sum(1 for pred, true in zip(always_negative_predictions, true_labels) if pred == true)
        accuracy = correct_predictions / total_samples

        print(f"总是预测负类的准确率: {accuracy:.1%}")
        print(f"但是召回率（检测到的正类）: {0:.1%}")

        return accuracy

    # accuracy_paradox_example()
    ```

    #### 2. **模型偏向多数类**
    大多数机器学习算法被设计为最小化整体错误率，这自然导致它们偏向于多数类。

    #### 3. **少数类学习不足**
    由于少数类样本太少，模型难以学习到足够的模式来准确识别这些类别。

    ### 类别不平衡的检测

    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter

    def analyze_class_imbalance(y, class_names=None):
        \"\"\"分析类别不平衡程度\"\"\"

        # 计算类别分布
        class_counts = Counter(y)
        total_samples = len(y)

        print("=== 类别不平衡分析 ===")
        print(f"总样本数: {total_samples:,}")
        print(f"类别数量: {len(class_counts)}")
        print()

        # 计算不平衡比率
        sorted_counts = sorted(class_counts.values(), reverse=True)
        majority_count = sorted_counts[0]
        minority_count = sorted_counts[-1]
        imbalance_ratio = majority_count / minority_count

        print(f"不平衡比率: {imbalance_ratio:.1f}:1")
        print(f"多数类占比: {majority_count/total_samples:.1%}")
        print(f"少数类占比: {minority_count/total_samples:.1%}")
        print()

        # 详细分布
        print("详细类别分布:")
        for class_label, count in sorted(class_counts.items()):
            percentage = count / total_samples * 100
            class_name = class_names[class_label] if class_names else f"类别 {class_label}"
            print(f"  {class_name}: {count:,} ({percentage:.1f}%)")

        # 不平衡严重程度评估
        if imbalance_ratio < 2:
            severity = "轻微不平衡"
        elif imbalance_ratio < 10:
            severity = "中等不平衡"
        elif imbalance_ratio < 100:
            severity = "严重不平衡"
        else:
            severity = "极度不平衡"

        print(f"\\n不平衡严重程度: {severity}")

        return {
            'imbalance_ratio': imbalance_ratio,
            'class_distribution': class_counts,
            'severity': severity
        }
    ```

    ### 处理类别不平衡的方法

    处理类别不平衡主要有两大类方法：**数据层面的方法**和**算法层面的方法**。

    #### 数据层面的方法

    ##### 1. **过采样（Oversampling）**

    增加少数类样本的数量：

    ```python
    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

    def oversampling_techniques(X, y):
        \"\"\"不同的过采样技术对比\"\"\"

        techniques = {
            'Random Oversampling': RandomOverSampler(random_state=42),
            'SMOTE': SMOTE(random_state=42),
            'ADASYN': ADASYN(random_state=42)
        }

        results = {}

        for name, sampler in techniques.items():
            try:
                X_resampled, y_resampled = sampler.fit_resample(X, y)

                results[name] = {
                    'original_shape': X.shape,
                    'resampled_shape': X_resampled.shape,
                    'original_distribution': Counter(y),
                    'resampled_distribution': Counter(y_resampled)
                }

                print(f"\\n{name}:")
                print(f"  原始数据: {X.shape[0]} 样本")
                print(f"  重采样后: {X_resampled.shape[0]} 样本")
                print(f"  原始分布: {dict(Counter(y))}")
                print(f"  重采样分布: {dict(Counter(y_resampled))}")

            except Exception as e:
                print(f"{name} 失败: {e}")
                results[name] = None

        return results
    ```

    **SMOTE（Synthetic Minority Oversampling Technique）**是最流行的过采样技术：

    ```python
    def smote_explanation():
        \"\"\"SMOTE算法原理解释\"\"\"

        print("=== SMOTE算法原理 ===")
        print("1. 对于每个少数类样本:")
        print("   - 找到k个最近邻（通常k=5）")
        print("   - 随机选择其中一个邻居")
        print("   - 在该样本和选定邻居之间的线段上随机生成新样本")
        print()
        print("2. 数学表示:")
        print("   new_sample = sample + λ × (neighbor - sample)")
        print("   其中 λ 是 [0,1] 之间的随机数")
        print()
        print("3. 优点:")
        print("   - 生成合理的合成样本")
        print("   - 不是简单复制，而是创造新的变化")
        print("   - 在特征空间中填补少数类区域")
        print()
        print("4. 缺点:")
        print("   - 可能在噪声区域生成样本")
        print("   - 对高维数据效果可能不佳")
        print("   - 可能导致过拟合")

    # smote_explanation()
    ```

    ##### 2. **欠采样（Undersampling）**

    减少多数类样本的数量：

    ```python
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours

    def undersampling_techniques(X, y):
        \"\"\"不同的欠采样技术\"\"\"

        techniques = {
            'Random Undersampling': RandomUnderSampler(random_state=42),
            'Tomek Links': TomekLinks(),
            'Edited Nearest Neighbours': EditedNearestNeighbours()
        }

        results = {}

        for name, sampler in techniques.items():
            try:
                X_resampled, y_resampled = sampler.fit_resample(X, y)

                results[name] = {
                    'data_reduction': 1 - (X_resampled.shape[0] / X.shape[0]),
                    'class_distribution': Counter(y_resampled)
                }

                print(f"\\n{name}:")
                print(f"  数据减少: {results[name]['data_reduction']:.1%}")
                print(f"  最终分布: {dict(Counter(y_resampled))}")

            except Exception as e:
                print(f"{name} 失败: {e}")
                results[name] = None

        return results
    ```

    ##### 3. **混合采样**

    结合过采样和欠采样：

    ```python
    from imblearn.combine import SMOTETomek, SMOTEENN

    def combined_sampling(X, y):
        \"\"\"混合采样技术\"\"\"

        techniques = {
            'SMOTE + Tomek': SMOTETomek(random_state=42),
            'SMOTE + ENN': SMOTEENN(random_state=42)
        }

        for name, sampler in techniques.items():
            X_resampled, y_resampled = sampler.fit_resample(X, y)

            print(f"\\n{name}:")
            print(f"  原始: {Counter(y)}")
            print(f"  处理后: {Counter(y_resampled)}")
            print(f"  样本变化: {X.shape[0]} → {X_resampled.shape[0]}")
    ```

    #### 算法层面的方法

    ##### 1. **类别权重调整**

    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.utils.class_weight import compute_class_weight

    def class_weight_methods(X, y):
        \"\"\"类别权重调整方法\"\"\"

        # 计算类别权重
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))

        print("计算的类别权重:")
        for class_label, weight in class_weight_dict.items():
            print(f"  类别 {class_label}: {weight:.2f}")

        # 不同算法的类别权重使用
        models = {
            'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
            'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
            'Custom Weights': LogisticRegression(class_weight=class_weight_dict, random_state=42)
        }

        return models, class_weight_dict
    ```

    ##### 2. **阈值调整**

    ```python
    from sklearn.metrics import precision_recall_curve, roc_curve
    import matplotlib.pyplot as plt

    def threshold_optimization(y_true, y_proba, metric='f1'):
        \"\"\"优化分类阈值\"\"\"

        # 计算不同阈值下的指标
        precision, recall, thresholds_pr = precision_recall_curve(y_true, y_proba)

        # 计算F1分数
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores)  # 处理除零情况

        # 找到最佳阈值
        if metric == 'f1':
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds_pr[best_idx]
            best_score = f1_scores[best_idx]

            print(f"最佳F1阈值: {best_threshold:.3f}")
            print(f"最佳F1分数: {best_score:.3f}")
            print(f"对应精确率: {precision[best_idx]:.3f}")
            print(f"对应召回率: {recall[best_idx]:.3f}")

        return best_threshold, best_score
    ```

    ##### 3. **集成方法**

    ```python
    from sklearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
    from imblearn.ensemble import EasyEnsembleClassifier

    def ensemble_methods_for_imbalance():
        \"\"\"专门处理不平衡数据的集成方法\"\"\"

        methods = {
            'Balanced Random Forest': BalancedRandomForestClassifier(random_state=42),
            'Balanced Bagging': BalancedBaggingClassifier(random_state=42),
            'Easy Ensemble': EasyEnsembleClassifier(random_state=42)
        }

        print("不平衡数据的集成方法:")
        for name, model in methods.items():
            print(f"\\n{name}:")
            print(f"  原理: {model.__class__.__doc__.split('.')[0] if model.__class__.__doc__ else '专门处理不平衡数据的集成方法'}")

        return methods
    ```

    ### 评估不平衡数据模型

    对于不平衡数据，准确率不是好的评估指标。应该使用：

    ```python
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

    def evaluate_imbalanced_model(y_true, y_pred, y_proba=None):
        \"\"\"评估不平衡数据模型的完整指标\"\"\"

        print("=== 不平衡数据模型评估 ===")

        # 1. 分类报告
        print("\\n1. 分类报告:")
        print(classification_report(y_true, y_pred))

        # 2. 混淆矩阵
        print("\\n2. 混淆矩阵:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)

        # 3. 关键指标计算
        tn, fp, fn, tp = cm.ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\\n3. 关键指标:")
        print(f"   精确率 (Precision): {precision:.3f}")
        print(f"   召回率 (Recall): {recall:.3f}")
        print(f"   特异性 (Specificity): {specificity:.3f}")
        print(f"   F1分数: {f1:.3f}")

        # 4. AUC指标
        if y_proba is not None:
            auc_roc = roc_auc_score(y_true, y_proba)
            auc_pr = average_precision_score(y_true, y_proba)

            print(f"\\n4. AUC指标:")
            print(f"   ROC-AUC: {auc_roc:.3f}")
            print(f"   PR-AUC: {auc_pr:.3f}")

        return {
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'auc_roc': auc_roc if y_proba is not None else None,
            'auc_pr': auc_pr if y_proba is not None else None
        }
    ```

    类别不平衡是现实世界ML项目中的常见挑战。选择合适的处理方法取决于你的具体问题、数据特征和业务需求。通常建议尝试多种方法并使用适当的评估指标来选择最佳方案。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 数据泄漏：ML中的隐形杀手

    数据泄漏是机器学习中最危险的陷阱之一。它会让你的模型在训练和验证时表现出色，但在生产环境中完全失效。理解和防止数据泄漏对于构建可靠的ML系统至关重要。

    ### 什么是数据泄漏？

    **数据泄漏**指的是训练数据中包含了在实际预测时不应该可用的信息。换句话说，模型"偷看"了未来的信息或目标变量的直接代理。

    ![数据泄漏概念](https://www.dailydoseofds.com/content/images/2025/08/image-170.png)

    ### 数据泄漏的类型

    #### 1. **时间泄漏（Temporal Leakage）**

    这是最常见的泄漏类型，发生在使用未来信息预测过去事件时。

    ```python
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    def temporal_leakage_example():
        \"\"\"时间泄漏示例\"\"\"

        # 创建时间序列数据
        dates = pd.date_range('2023-01-01', periods=1000, freq='D')

        # 模拟股票价格数据
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)

        df = pd.DataFrame({
            'date': dates,
            'price': prices,
            'volume': np.random.randint(1000, 10000, 1000)
        })

        # ❌ 错误：使用未来7天的平均价格作为特征
        df['future_7day_avg'] = df['price'].rolling(window=7, center=True).mean()

        # ❌ 错误：使用全局统计信息
        df['price_zscore_global'] = (df['price'] - df['price'].mean()) / df['price'].std()

        # ✅ 正确：只使用历史信息
        df['past_7day_avg'] = df['price'].rolling(window=7, min_periods=1).mean().shift(1)
        df['price_zscore_rolling'] = df['price'].rolling(window=30, min_periods=1).apply(
            lambda x: (x.iloc[-1] - x.mean()) / x.std() if len(x) > 1 else 0
        ).shift(1)

        print("时间泄漏示例:")
        print("❌ 错误特征:")
        print("  - future_7day_avg: 使用了未来信息")
        print("  - price_zscore_global: 使用了全局统计信息")
        print("✅ 正确特征:")
        print("  - past_7day_avg: 只使用历史信息")
        print("  - price_zscore_rolling: 使用滚动窗口统计")

        return df

    # temporal_leakage_example()
    ```

    #### 2. **目标泄漏（Target Leakage）**

    特征直接包含目标变量的信息或其强代理。

    ```python
    def target_leakage_examples():
        \"\"\"目标泄漏示例\"\"\"

        examples = {
            "信用卡欺诈检测": {
                "❌ 泄漏特征": [
                    "transaction_flagged_by_bank",  # 银行已经标记为可疑
                    "customer_account_frozen",      # 账户被冻结（结果的直接指示）
                    "fraud_investigation_opened"    # 已开始欺诈调查
                ],
                "✅ 正确特征": [
                    "transaction_amount",
                    "merchant_category",
                    "time_of_day",
                    "days_since_last_transaction"
                ]
            },

            "医疗诊断": {
                "❌ 泄漏特征": [
                    "prescribed_medication",        # 处方药物（诊断后的结果）
                    "specialist_referral",          # 专科转诊（诊断的结果）
                    "treatment_plan"               # 治疗计划（诊断后制定）
                ],
                "✅ 正确特征": [
                    "patient_symptoms",
                    "vital_signs",
                    "medical_history",
                    "lab_test_results"
                ]
            },

            "客户流失预测": {
                "❌ 泄漏特征": [
                    "account_closure_date",         # 账户关闭日期
                    "final_bill_amount",           # 最终账单金额
                    "retention_call_made"          # 挽留电话（流失后的行动）
                ],
                "✅ 正确特征": [
                    "monthly_usage_trend",
                    "customer_service_calls",
                    "payment_history",
                    "contract_length"
                ]
            }
        }

        for scenario, features in examples.items():
            print(f"\\n=== {scenario} ===")
            print("❌ 泄漏特征（不应使用）:")
            for feature in features["❌ 泄漏特征"]:
                print(f"  - {feature}")
            print("✅ 正确特征（可以使用）:")
            for feature in features["✅ 正确特征"]:
                print(f"  - {feature}")

        return examples

    # target_leakage_examples()
    ```

    #### 3. **训练-测试污染（Train-Test Contamination）**

    在数据分割之前进行预处理，导致测试集信息泄漏到训练集。

    ```python
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    def train_test_contamination_demo():
        \"\"\"训练-测试污染演示\"\"\"

        # 生成示例数据
        np.random.seed(42)
        X = np.random.randn(1000, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        print("=== 训练-测试污染演示 ===\\n")

        # ❌ 错误方法：先标准化，后分割
        print("❌ 错误方法：先标准化，后分割")
        scaler_wrong = StandardScaler()
        X_scaled_wrong = scaler_wrong.fit_transform(X)  # 使用全部数据计算统计信息
        X_train_wrong, X_test_wrong, y_train, y_test = train_test_split(
            X_scaled_wrong, y, test_size=0.2, random_state=42
        )

        print(f"训练集均值: {X_train_wrong.mean(axis=0)}")
        print(f"测试集均值: {X_test_wrong.mean(axis=0)}")
        print("问题：测试集的统计信息已经泄漏到标准化过程中\\n")

        # ✅ 正确方法：先分割，后标准化
        print("✅ 正确方法：先分割，后标准化")
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler_correct = StandardScaler()
        X_train_correct = scaler_correct.fit_transform(X_train_raw)  # 只用训练集计算统计信息
        X_test_correct = scaler_correct.transform(X_test_raw)        # 用训练集统计信息转换测试集

        print(f"训练集均值: {X_train_correct.mean(axis=0)}")
        print(f"测试集均值: {X_test_correct.mean(axis=0)}")
        print("正确：测试集使用训练集的统计信息进行转换")

        return {
            'wrong_method': (X_train_wrong, X_test_wrong),
            'correct_method': (X_train_correct, X_test_correct)
        }

    # train_test_contamination_demo()
    ```

    #### 4. **数据收集/标注过程中的泄漏**

    数据收集或标注过程中引入的偏差。

    ```python
    def data_collection_leakage_examples():
        \"\"\"数据收集过程中的泄漏示例\"\"\"

        examples = [
            {
                "场景": "医学影像诊断",
                "问题": "疾病患者的X光片来自特定医院，健康人的X光片来自另一医院",
                "泄漏": "模型学会了识别医院标记而不是疾病特征",
                "解决方案": "确保正负样本来自相同的数据源和设备"
            },
            {
                "场景": "COVID-19检测",
                "问题": "阳性病例主要来自某些医院，阴性病例来自其他医院",
                "泄漏": "模型学会了医院来源而不是实际患者数据",
                "解决方案": "平衡不同来源的正负样本分布"
            },
            {
                "场景": "文本分类",
                "问题": "不同类别的文本在不同时间收集，包含时间戳信息",
                "泄漏": "模型可能学会了时间模式而不是文本内容",
                "解决方案": "移除或随机化时间相关的标识符"
            }
        ]

        print("=== 数据收集过程中的泄漏示例 ===\\n")

        for i, example in enumerate(examples, 1):
            print(f"{i}. {example['场景']}")
            print(f"   问题: {example['问题']}")
            print(f"   泄漏: {example['泄漏']}")
            print(f"   解决方案: {example['解决方案']}\\n")

        return examples

    # data_collection_leakage_examples()
    ```

    ### 检测数据泄漏

    #### 1. **性能异常检测**

    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score

    def detect_leakage_by_performance(X_train, X_test, y_train, y_test):
        \"\"\"通过异常高的性能检测数据泄漏\"\"\"

        # 训练简单模型
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 评估性能
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        print("=== 性能异常检测 ===")
        print(f"训练准确率: {train_accuracy:.4f}")
        print(f"测试准确率: {test_accuracy:.4f}")
        print(f"性能差异: {abs(train_accuracy - test_accuracy):.4f}")

        # 泄漏检测规则
        if test_accuracy > 0.95:
            print("⚠️  警告：测试准确率异常高，可能存在数据泄漏")

        if abs(train_accuracy - test_accuracy) < 0.01 and test_accuracy > 0.9:
            print("⚠️  警告：训练和测试性能过于接近，可能存在泄漏")

        return train_accuracy, test_accuracy
    ```

    #### 2. **特征重要性分析**

    ```python
    def analyze_feature_importance_for_leakage(model, feature_names, threshold=0.5):
        \"\"\"通过特征重要性分析检测泄漏\"\"\"

        # 获取特征重要性
        importances = model.feature_importances_

        # 创建特征重要性DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("=== 特征重要性分析 ===")
        print("前10个最重要的特征:")
        print(feature_importance_df.head(10))

        # 检测异常重要的特征
        max_importance = importances.max()
        if max_importance > threshold:
            dominant_feature = feature_importance_df.iloc[0]['feature']
            print(f"\\n⚠️  警告：特征 '{dominant_feature}' 重要性异常高 ({max_importance:.3f})")
            print("   这可能表明存在数据泄漏")

        # 检测重要性分布
        top_3_importance = feature_importance_df.head(3)['importance'].sum()
        if top_3_importance > 0.8:
            print("\\n⚠️  警告：前3个特征占据了过高的重要性")
            print("   建议检查这些特征是否包含目标信息")

        return feature_importance_df
    ```

    #### 3. **时间一致性检查**

    ```python
    def temporal_consistency_check(data, time_column, target_column):
        \"\"\"时间一致性检查\"\"\"

        print("=== 时间一致性检查 ===")

        # 检查未来信息
        data_sorted = data.sort_values(time_column)

        # 检查是否有特征使用了未来信息
        suspicious_features = []

        for col in data.columns:
            if col not in [time_column, target_column]:
                # 检查特征值是否与未来目标值高度相关
                correlation_with_future = data_sorted[col].corr(
                    data_sorted[target_column].shift(-1)  # 未来目标值
                )

                if abs(correlation_with_future) > 0.7:
                    suspicious_features.append((col, correlation_with_future))

        if suspicious_features:
            print("⚠️  发现可疑特征（与未来目标值高度相关）:")
            for feature, corr in suspicious_features:
                print(f"   {feature}: 相关性 = {corr:.3f}")
        else:
            print("✅ 未发现明显的时间泄漏")

        return suspicious_features
    ```

    ### 防止数据泄漏的最佳实践

    #### 1. **严格的数据分割流程**

    ```python
    def leakage_safe_pipeline():
        \"\"\"防泄漏的安全管道\"\"\"

        pipeline_steps = [
            "1. 数据收集和初步清理",
            "2. 早期数据分割（训练/验证/测试）",
            "3. 只在训练集上进行特征工程",
            "4. 只在训练集上计算统计信息",
            "5. 将训练集的转换应用到验证/测试集",
            "6. 模型训练（只使用训练集）",
            "7. 超参数调优（使用验证集）",
            "8. 最终评估（使用测试集，只评估一次）"
        ]

        print("=== 防泄漏的安全管道 ===")
        for step in pipeline_steps:
            print(f"  {step}")

        print("\\n关键原则:")
        print("  - 测试集是神圣的：永远不要用于训练或调优")
        print("  - 时间顺序：对于时间序列，严格按时间分割")
        print("  - 统计信息：只使用训练集计算均值、方差等")
        print("  - 特征工程：在分割后进行，避免信息泄漏")

        return pipeline_steps
    ```

    #### 2. **自动化泄漏检测**

    ```python
    class LeakageDetector:
        \"\"\"自动化数据泄漏检测器\"\"\"

        def __init__(self):
            self.checks = []
            self.warnings = []

        def add_performance_check(self, train_score, test_score, threshold=0.95):
            \"\"\"添加性能检查\"\"\"
            if test_score > threshold:
                self.warnings.append(f"测试性能异常高: {test_score:.3f}")

            if abs(train_score - test_score) < 0.01 and test_score > 0.9:
                self.warnings.append("训练和测试性能过于接近")

        def add_feature_importance_check(self, importances, feature_names, threshold=0.5):
            \"\"\"添加特征重要性检查\"\"\"
            max_idx = np.argmax(importances)
            max_importance = importances[max_idx]

            if max_importance > threshold:
                feature_name = feature_names[max_idx]
                self.warnings.append(f"特征 '{feature_name}' 重要性异常高: {max_importance:.3f}")

        def add_temporal_check(self, data, time_col, target_col):
            \"\"\"添加时间一致性检查\"\"\"
            # 检查是否有特征名包含"future"等关键词
            future_keywords = ['future', 'next', 'after', 'post', 'following']

            for col in data.columns:
                if any(keyword in col.lower() for keyword in future_keywords):
                    self.warnings.append(f"特征名可疑: '{col}' 可能包含未来信息")

        def generate_report(self):
            \"\"\"生成泄漏检测报告\"\"\"
            print("=== 数据泄漏检测报告 ===")

            if not self.warnings:
                print("✅ 未发现明显的数据泄漏迹象")
            else:
                print(f"⚠️  发现 {len(self.warnings)} 个潜在问题:")
                for i, warning in enumerate(self.warnings, 1):
                    print(f"   {i}. {warning}")

            print("\\n建议:")
            print("  - 仔细检查标记的特征")
            print("  - 验证数据分割流程")
            print("  - 确认特征工程的时间正确性")
            print("  - 考虑使用时间基础的验证")

            return len(self.warnings) == 0
    ```

    数据泄漏是ML项目失败的主要原因之一。通过理解不同类型的泄漏、实施检测机制和遵循最佳实践，你可以构建真正可靠的机器学习系统。记住：**在预测时刻，只使用那时真正可用的数据**。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 大规模特征存储（Feast）

    随着机器学习系统的成熟，组织发现自己在模型之间重用特征，并需要训练和服务的一致数据。特征存储应运而生来解决这个问题。

    **特征存储**是ML特征的集中数据存储和管理系统。Feast是一个流行的开源特征存储。

    ![特征存储架构](https://www.dailydoseofds.com/content/images/2025/08/image-180.png)

    ### 为什么需要特征存储？

    在生产ML系统中，你可能有许多模型使用重叠的特征。如果每个管道独立计算特征，你会得到重复、不一致和维护头痛。

    **特征存储创建了单一的真实来源**：计算特征X一次，在任何地方使用它。它还提供在线查找，所以你的服务代码可以用低延迟按键查询特征。这避免了在生产请求中即时重新计算特征，这可能太慢或不一致。

    #### 特征存储解决的核心问题

    ```python
    def feature_store_problems():
        \"\"\"特征存储解决的核心问题\"\"\"

        problems = {
            "特征重复计算": {
                "问题": "多个团队重复实现相同的特征逻辑",
                "后果": "浪费计算资源，维护成本高，结果不一致",
                "解决方案": "集中化特征定义和计算"
            },

            "训练-服务偏差": {
                "问题": "训练时的特征计算与服务时不同",
                "后果": "模型在生产中性能下降",
                "解决方案": "统一的特征定义确保一致性"
            },

            "特征发现困难": {
                "问题": "团队不知道已有哪些特征可用",
                "后果": "重复工作，错失有价值的特征",
                "解决方案": "特征注册表和文档"
            },

            "数据泄漏风险": {
                "问题": "特征计算中使用了未来信息",
                "后果": "模型在生产中失效",
                "解决方案": "时间点正确性保证"
            },

            "实时特征服务": {
                "问题": "生产环境需要低延迟特征查询",
                "后果": "用户体验差，系统响应慢",
                "解决方案": "在线特征存储"
            }
        }

        print("=== 特征存储解决的核心问题 ===\\n")

        for problem, details in problems.items():
            print(f"📋 {problem}")
            print(f"   问题: {details['问题']}")
            print(f"   后果: {details['后果']}")
            print(f"   解决方案: {details['解决方案']}\\n")

        return problems

    # feature_store_problems()
    ```

    ### Feast架构概述

    Feast将特征计算和存储解耦。我们在特征仓库中定义特征。

    #### 核心概念

    ```python
    def feast_core_concepts():
        \"\"\"Feast核心概念解释\"\"\"

        concepts = {
            "Entity（实体）": {
                "定义": "特征的主键，具有值类型",
                "示例": "customer_id, product_id, user_id",
                "作用": "用于特征查询和连接的唯一标识符"
            },

            "Feature View（特征视图）": {
                "定义": "一组特征的定义（具有特定模式和实体）以及如何获取数据",
                "示例": "用户行为特征、产品统计特征",
                "作用": "定义特征的计算逻辑和数据源"
            },

            "Offline Store（离线存储）": {
                "定义": "存储历史特征数据的地方",
                "示例": "BigQuery, Redshift, 文件系统",
                "作用": "用于训练数据生成和批量特征计算"
            },

            "Online Store（在线存储）": {
                "定义": "用于服务特征给模型的快速键值存储",
                "示例": "Redis, DynamoDB, SQLite",
                "作用": "低延迟的实时特征查询"
            },

            "Feature Service（特征服务）": {
                "定义": "为方便检索而分组的特征",
                "示例": "推荐系统特征包、风控特征包",
                "作用": "简化特征查询和管理"
            }
        }

        print("=== Feast核心概念 ===\\n")

        for concept, details in concepts.items():
            print(f"🔧 {concept}")
            print(f"   定义: {details['定义']}")
            print(f"   示例: {details['示例']}")
            print(f"   作用: {details['作用']}\\n")

        return concepts

    # feast_core_concepts()
    ```

    ### Feast工作流程

    ```python
    def feast_workflow():
        \"\"\"Feast典型工作流程\"\"\"

        workflow_steps = [
            {
                "步骤": "1. 定义特征",
                "描述": "在Python代码中定义实体、特征视图和数据源",
                "代码示例": '''
    from feast import Entity, FeatureView, Field
    from feast.types import Float32, Int64

    # 定义实体
    customer = Entity(name="customer_id", value_type=ValueType.INT64)

    # 定义特征视图
    customer_features = FeatureView(
        name="customer_features",
        entities=["customer_id"],
        schema=[
            Field(name="age", dtype=Float32),
            Field(name="income", dtype=Float32),
        ],
        source=FileSource(path="customer_data.parquet")
    )
                '''
            },

            {
                "步骤": "2. 应用配置",
                "描述": "将特征定义部署到Feast注册表",
                "代码示例": "feast apply"
            },

            {
                "步骤": "3. 物化特征",
                "描述": "将特征数据从离线存储加载到在线存储",
                "代码示例": "feast materialize-incremental $(date)"
            },

            {
                "步骤": "4. 生成训练数据",
                "描述": "创建时间点正确的训练数据集",
                "代码示例": '''
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=["customer_features:age", "customer_features:income"]
    ).to_df()
                '''
            },

            {
                "步骤": "5. 在线特征服务",
                "描述": "在生产中查询实时特征",
                "代码示例": '''
    online_features = store.get_online_features(
        features=["customer_features:age", "customer_features:income"],
        entity_rows=[{"customer_id": 123}]
    ).to_dict()
                '''
            }
        ]

        print("=== Feast典型工作流程 ===\\n")

        for step in workflow_steps:
            print(f"📋 {step['步骤']}")
            print(f"   描述: {step['描述']}")
            if 'feast' in step['代码示例'] and len(step['代码示例']) < 50:
                print(f"   命令: {step['代码示例']}")
            else:
                print(f"   代码示例:")
                print(f"   ```python{step['代码示例']}   ```")
            print()

        return workflow_steps

    # feast_workflow()
    ```

    ### 实际Feast实现示例

    让我们通过一个完整的客户流失预测示例来演示Feast的使用：

    ```python
    # 注意：这是演示代码，实际运行需要安装Feast
    def feast_implementation_example():
        \"\"\"Feast实现示例（演示代码）\"\"\"

        print("=== Feast实现示例 ===\\n")

        # 1. 安装和设置
        setup_code = '''
    # 安装Feast
    pip install feast[redis]

    # 初始化Feast项目
    feast init customer_churn_project
    cd customer_churn_project
        '''

        print("1. 项目设置:")
        print(setup_code)

        # 2. 定义特征
        feature_definition = '''
    # feature_store.py
    from feast import Entity, FeatureView, Field, FileSource, FeatureStore
    from feast.types import Float32, Int64, String
    from datetime import timedelta

    # 定义实体
    customer = Entity(
        name="customer_id",
        value_type=ValueType.STRING,
        description="客户唯一标识符"
    )

    # 定义客户基础特征视图
    customer_demographics = FeatureView(
        name="customer_demographics",
        entities=["customer_id"],
        ttl=timedelta(days=365),
        schema=[
            Field(name="age", dtype=Int64),
            Field(name="gender", dtype=String),
            Field(name="income", dtype=Float32),
            Field(name="tenure_months", dtype=Int64),
        ],
        source=FileSource(
            path="data/customer_demographics.parquet",
            timestamp_field="event_timestamp"
        ),
        tags={"team": "ml", "domain": "customer"}
    )

    # 定义客户行为特征视图
    customer_behavior = FeatureView(
        name="customer_behavior",
        entities=["customer_id"],
        ttl=timedelta(days=30),
        schema=[
            Field(name="monthly_charges", dtype=Float32),
            Field(name="total_charges", dtype=Float32),
            Field(name="support_calls_30d", dtype=Int64),
            Field(name="login_frequency_30d", dtype=Int64),
        ],
        source=FileSource(
            path="data/customer_behavior.parquet",
            timestamp_field="event_timestamp"
        ),
        tags={"team": "ml", "domain": "behavior"}
    )

    # 定义特征服务
    churn_prediction_service = FeatureService(
        name="churn_prediction",
        features=[
            "customer_demographics:age",
            "customer_demographics:gender",
            "customer_demographics:income",
            "customer_demographics:tenure_months",
            "customer_behavior:monthly_charges",
            "customer_behavior:total_charges",
            "customer_behavior:support_calls_30d",
            "customer_behavior:login_frequency_30d",
        ]
    )
        '''

        print("2. 特征定义:")
        print(feature_definition)

        # 3. 训练数据生成
        training_code = '''
    # 生成训练数据
    from feast import FeatureStore
    import pandas as pd

    store = FeatureStore(repo_path=".")

    # 准备实体DataFrame（包含客户ID、时间戳和标签）
    entity_df = pd.DataFrame({
        "customer_id": ["CUST_001", "CUST_002", "CUST_003"],
        "event_timestamp": pd.to_datetime([
            "2023-01-01", "2023-01-02", "2023-01-03"
        ]),
        "churn_label": [0, 1, 0]  # 目标变量
    })

    # 获取历史特征（时间点正确）
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "customer_demographics:age",
            "customer_demographics:income",
            "customer_behavior:monthly_charges",
            "customer_behavior:support_calls_30d"
        ]
    ).to_df()

    print("训练数据生成完成")
    print(training_df.head())
        '''

        print("3. 训练数据生成:")
        print(training_code)

        # 4. 在线服务
        serving_code = '''
    # 物化特征到在线存储
    from datetime import datetime

    # 物化最新特征
    store.materialize_incremental(end_date=datetime.now())

    # 在线特征查询
    online_features = store.get_online_features(
        features=[
            "customer_demographics:age",
            "customer_demographics:income",
            "customer_behavior:monthly_charges",
            "customer_behavior:support_calls_30d"
        ],
        entity_rows=[
            {"customer_id": "CUST_001"},
            {"customer_id": "CUST_002"}
        ]
    )

    # 转换为字典格式
    feature_dict = online_features.to_dict()
    print("在线特征查询结果:")
    print(feature_dict)
        '''

        print("4. 在线特征服务:")
        print(serving_code)

        return {
            'setup': setup_code,
            'features': feature_definition,
            'training': training_code,
            'serving': serving_code
        }

    # feast_implementation_example()
    ```

    ### Feast的优势

    ```python
    def feast_advantages():
        \"\"\"Feast的主要优势\"\"\"

        advantages = {
            "时间点正确性": {
                "描述": "确保训练数据中的特征值是历史上该时间点真实可用的",
                "价值": "防止数据泄漏，确保模型在生产中的可靠性",
                "实现": "自动处理时间戳，确保特征-标签对齐"
            },

            "训练-服务一致性": {
                "描述": "训练和服务使用相同的特征定义和计算逻辑",
                "价值": "消除训练-服务偏差，提高模型生产性能",
                "实现": "统一的特征视图定义"
            },

            "特征重用": {
                "描述": "一次定义，多处使用的特征管理",
                "价值": "减少重复工作，提高开发效率",
                "实现": "集中化的特征注册表"
            },

            "可扩展性": {
                "描述": "支持从小规模到企业级的特征管理",
                "价值": "随业务增长而扩展",
                "实现": "灵活的存储后端选择"
            },

            "版本控制": {
                "描述": "特征定义的版本管理和回滚能力",
                "价值": "支持实验和安全部署",
                "实现": "Git集成和特征版本跟踪"
            }
        }

        print("=== Feast的主要优势 ===\\n")

        for advantage, details in advantages.items():
            print(f"🚀 {advantage}")
            print(f"   描述: {details['描述']}")
            print(f"   价值: {details['价值']}")
            print(f"   实现: {details['实现']}\\n")

        return advantages

    # feast_advantages()
    ```

    ### 特征存储最佳实践

    ```python
    def feature_store_best_practices():
        \"\"\"特征存储最佳实践\"\"\"

        practices = {
            "特征命名规范": [
                "使用描述性名称：customer_age_years 而不是 age",
                "包含时间窗口：purchases_30d, clicks_7d",
                "使用一致的命名约定：snake_case",
                "避免缩写：monthly_revenue 而不是 mon_rev"
            ],

            "特征组织": [
                "按业务域分组：customer_features, product_features",
                "按更新频率分组：daily_features, realtime_features",
                "按数据源分组：database_features, api_features",
                "使用标签进行分类和搜索"
            ],

            "数据质量": [
                "实施特征验证：数据类型、范围检查",
                "监控特征分布变化：数据漂移检测",
                "设置数据质量警报：缺失值、异常值",
                "定期审查特征使用情况"
            ],

            "性能优化": [
                "选择合适的TTL：平衡新鲜度和存储成本",
                "优化批处理窗口：减少计算开销",
                "使用适当的分区策略：提高查询性能",
                "监控存储和计算成本"
            ],

            "安全和治理": [
                "实施访问控制：基于角色的特征访问",
                "数据血缘跟踪：了解特征来源和依赖",
                "合规性检查：确保符合数据保护法规",
                "审计日志：跟踪特征使用和修改"
            ]
        }

        print("=== 特征存储最佳实践 ===\\n")

        for category, items in practices.items():
            print(f"📋 {category}")
            for item in items:
                print(f"   • {item}")
            print()

        return practices

    # feature_store_best_practices()
    ```

    特征存储如Feast为ML系统带来了组织性和可靠性。它们确保创建特征的辛勤工作不会在训练和服务之间被重复或损坏。通过提供时间点正确性、训练-服务一致性和特征重用，特征存储成为现代MLOps架构的关键组件。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 结论

    在本章中，我们通过关注生产就绪ML工作流的四个关键概念，扩展了对数据管道的探索：**采样策略、类别不平衡处理、数据泄漏防护和特征存储**。

    ### 🎯 **关键学习成果**

    #### 📊 **采样策略掌握**
    我们研究了采样技术的全谱，包括概率和非概率方法：

    - **概率采样**：简单随机、分层、系统和聚类采样的原理和应用
    - **非概率采样**：便利、雪球和判断采样的使用场景和限制
    - **ML特定考虑**：时间序列数据的时间基础分割、大数据的分块采样
    - **质量评估**：采样质量的量化评估方法

    **核心洞察**：正确的采样策略是模型泛化能力的基础，错误的采样可能导致严重的偏差和生产失败。

    #### ⚖️ **类别不平衡处理精通**
    我们深入研究了数据和算法层面的方法，如重采样、类别权重和焦点损失，构建优先考虑真实世界影响而非表面准确率的模型：

    - **数据层面方法**：
      - 过采样技术（SMOTE、ADASYN）的原理和实现
      - 欠采样方法（随机、Tomek Links、ENN）的应用
      - 混合采样策略的优势

    - **算法层面方法**：
      - 类别权重调整的数学原理
      - 阈值优化技术
      - 专门的集成方法

    - **评估策略**：超越准确率的综合评估指标体系

    **核心洞察**：类别不平衡是现实世界的常态，需要专门的技术和评估方法来确保模型的实用性。

    #### 🛡️ **数据泄漏防护专业知识**
    我们剖析了数据泄漏的陷阱，理解了微妙的疏忽如何使整个管道失效：

    - **泄漏类型识别**：
      - 时间泄漏：使用未来信息的危险
      - 目标泄漏：特征中包含目标信息
      - 训练-测试污染：预处理顺序的重要性
      - 数据收集泄漏：源头偏差的影响

    - **检测机制**：
      - 性能异常检测：识别"太好"的结果
      - 特征重要性分析：发现可疑的主导特征
      - 时间一致性检查：验证时间逻辑
      - 自动化检测系统：持续监控

    - **防护最佳实践**：严格的数据分割流程和"预测时刻正确性"原则

    **核心洞察**：数据泄漏是ML项目失败的隐形杀手，预防胜于治疗，系统性的检测和防护机制是必需的。

    #### 🏗️ **特征存储架构理解**
    我们将这些经验教训集中在Feast的实际演练中，展示了特征存储如何作为防泄漏、一致和可扩展ML管道的骨干：

    - **架构组件**：
      - 实体、特征视图、离线/在线存储的作用
      - 特征服务的组织和管理
      - 时间点正确性的技术实现

    - **工作流程**：
      - 从特征定义到生产服务的完整流程
      - 训练数据生成的最佳实践
      - 在线特征查询的优化

    - **企业价值**：
      - 特征重用和一致性保证
      - 开发效率的显著提升
      - 训练-服务偏差的消除

    **核心洞察**：特征存储不仅是技术工具，更是组织ML能力的战略资产。

    ### 💡 **系统性思维的体现**

    #### 🔄 **端到端一致性**
    本章强调了ML系统中各个组件的相互依赖性：

    - **采样决策**影响模型的泛化能力
    - **不平衡处理**影响业务价值的实现
    - **泄漏防护**影响生产可靠性
    - **特征存储**影响整体系统的可维护性

    #### 📈 **质量优先的理念**
    我们看到了数据质量如何贯穿整个ML生命周期：

    - **采样质量**决定了数据的代表性
    - **平衡处理**确保了模型的公平性
    - **泄漏防护**保证了结果的真实性
    - **特征管理**维护了系统的一致性

    #### 🎯 **业务价值导向**
    每个技术决策都与业务成果紧密相关：

    - 正确的采样策略确保模型在真实场景中有效
    - 适当的不平衡处理优化业务关键指标
    - 严格的泄漏防护避免生产灾难
    - 高效的特征管理加速产品迭代

    ### 🚀 **关键要点总结**

    **本章的关键要点是：数据设计选择——采样、不平衡处理和泄漏防护——与模型本身一样关键。** 像Feast这样的特征存储将这些实践从临时修复提升为系统级保证，确保大规模的可靠ML。

    这个原则体现在几个层面：

    #### 🔧 **技术层面**
    - 系统化的方法论胜过临时的解决方案
    - 自动化的检测和防护机制是必需的
    - 工具和流程的标准化提高了可靠性

    #### 👥 **组织层面**
    - 跨团队的特征共享提高了效率
    - 统一的最佳实践减少了错误
    - 知识的系统化传承加速了团队成长

    #### 💼 **业务层面**
    - 可靠的ML系统支撑业务决策
    - 高质量的数据管道创造竞争优势
    - 系统性的方法降低了运营风险

    ### 🔮 **未来展望**

    在下一部分中，我们将继续深入MLOps周期数据阶段本身的更多高级概念和工具。

    在数据阶段之后，我们将继续这个速成课程的旅程：

    #### 🔄 **CI/CD工作流**
    - 为ML系统量身定制的持续集成和部署
    - 自动化测试和质量保证
    - 模型部署的最佳实践

    #### 🏢 **行业案例研究**
    - 来自行业的真实世界案例研究
    - 不同规模和领域的成功模式
    - 失败案例的经验教训

    #### 🤖 **模型开发和实践**
    - 模型训练和验证的高级技术
    - 超参数优化和AutoML
    - 模型解释性和可信AI

    #### 📊 **生产监控和观察**
    - 模型性能的持续监控
    - 数据和模型漂移的检测
    - 异常检测和自动响应

    #### 🧠 **LLMOps特殊考虑**
    - 大语言模型的特殊运营需求
    - 提示工程和版本控制
    - 成本优化和性能调优

    #### 🔗 **完整端到端示例**
    - 结合生命周期所有元素的综合案例
    - 从数据到部署的完整工作流
    - 企业级MLOps平台的架构设计

    ### 🎪 **最终目标**

    目标，一如既往，是帮助你培养成熟的、**以系统为中心的思维方式**，将机器学习不视为独立的工件，而是更广泛软件生态系统的活跃部分。

    通过掌握这些高级数据工程概念，你已经具备了：

    - **系统性思维**：理解各组件间的相互作用
    - **质量意识**：优先考虑数据质量和系统可靠性
    - **业务导向**：将技术决策与业务价值对齐
    - **前瞻性规划**：设计可扩展和可维护的系统
    - **风险管理**：识别和防范潜在的系统性风险

    这些能力将使你能够构建真正企业级的ML系统，不仅在实验室中表现出色，更能在复杂的生产环境中创造持续的业务价值。

    ---

    🚀 **继续你的MLOps精进之旅，记住：优秀的数据工程是可靠ML系统的基石，而系统性的方法是长期成功的保证！**
    """
    )
    return


if __name__ == "__main__":
    app.run()
