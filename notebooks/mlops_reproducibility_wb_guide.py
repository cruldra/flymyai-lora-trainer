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
    # 完整的MLOps蓝图：ML系统中的可重现性和版本控制—第B部分（含实现）

    MLOps和LLMOps速成课程—第4部分

    ## 回顾

    在这个MLOps和LLMOps速成课程的第3部分中，我们通过探索可重现性和版本控制的重要性，加深了对ML系统的理解。

    ![MLOps概览](https://www.dailydoseofds.com/content/images/2025/08/image-74.png)

    我们首先探索了什么是可重现性，版本控制如何在实现可重现性中发挥关键作用，以及为什么这些概念首先很重要。

    ![可重现性重要性](https://www.dailydoseofds.com/content/images/2025/08/image-75.png)

    我们研究了可重现性在错误跟踪、协作、法规合规和生产环境等领域的重要性。

    ![协作问题](https://www.dailydoseofds.com/content/images/2025/08/image-76.png)
    *由于缺乏可重现性导致的协作问题*

    然后我们讨论了一些可能阻碍可重现性的主要挑战。我们看到ML是"部分代码，部分数据"如何增加了额外的复杂性层次。

    ![ML复杂性](https://www.dailydoseofds.com/content/images/2025/08/image-77.png)

    之后，我们回顾了确保ML项目和系统中可重现性和版本控制的最佳实践，包括代码和数据版本控制、维护过程确定性、实验跟踪和环境管理。

    ![最佳实践](https://www.dailydoseofds.com/content/images/2025/08/image-79.png)

    最后，我们通过涵盖种子固定、使用DVC进行数据版本控制和使用MLflow进行实验跟踪的实际模拟进行了演练。

    ![实践演示](https://www.dailydoseofds.com/content/images/2025/08/image-78.png)

    如果你还没有探索第3部分，我们强烈建议先阅读它，因为它奠定了概念框架和实现理解，这将帮助你更好地理解我们即将深入的内容。

    在本章中，我们将继续讨论ML系统中的可重现性和版本控制，深入探讨实际实现。

    我们将具体看到如何使用Weights & Biases (W&B)作为主要工具在ML项目中实现可重现性和版本控制，并比较W&B的方法与DVC和MLflow的方法。

    以W&B为核心的实现，我们将涵盖：

    - **实验跟踪**
    - **数据集和模型版本控制**
    - **可重现的管道**
    - **模型注册表**

    一如既往，每个概念都将得到具体示例、演练和实用技巧的支持，帮助你掌握想法和实现。

    让我们开始吧！

    ---

    ## 引言

    在我们深入Weights & Biases (W&B)的细节之前，让我们快速回顾与可重现性和版本控制相关的核心思想。

    正如在这个速成课程中多次讨论的，机器学习项目不会以构建在单次训练运行中表现良好的模型而结束。

    在机器学习系统中，如我们所知，我们不仅有代码，还有数据、模型、超参数、训练配置和环境依赖。

    ![ML系统组件](https://www.dailydoseofds.com/content/images/2025/08/image-82.png)

    确保可重现性意味着你获得的任何结果都可以在以后一致地重现，给定相同的输入（代码、数据、配置等）。

    ![可重现性定义](https://www.dailydoseofds.com/content/images/2025/08/image-86.png)

    通过系统地记录这些，你可以实现所谓的实验可重现性和审计。团队可以验证彼此的结果，在知道它们处于平等基础上的情况下比较实验，并在需要时回滚到以前的模型或数据集。

    ML中的版本控制与可重现性密切相关。我们不仅需要代码的版本控制，还需要数据集和模型的版本控制：

    - **数据集版本控制**：想象你的数据集用新样本或改进的标签进行了更新。在上一章中，我们为此目的查看了DVC。在本章中，我们将专注于W&B Artifacts，它也允许你将数据集作为版本化资产进行管理，类似于代码的版本控制方式。

    ![数据集版本控制](https://www.dailydoseofds.com/content/images/2025/08/image-85.png)

    - **模型版本控制**：同样，你可能会多次重新训练模型。模型注册表和版本控制机制让你跟踪不同的模型检查点（例如，"模型v1.0 vs v1.1"）以及它们的评估指标。这确保如果新部署出现问题，你总是可以回滚到以前的模型。

    ![模型版本控制](https://www.dailydoseofds.com/content/images/2025/08/image-84.png)

    总结一下，MLOps中的可重现性和版本控制是关于为实验混乱带来秩序：

    - 它们确保你可以通过记录进入实验的所有内容来完全重复任何实验
    - 它们允许你苹果对苹果地比较实验，因为你知道它们之间在代码/数据/配置方面的精确差异

    ![实验比较](https://www.dailydoseofds.com/content/images/2025/08/image-83.png)

    - 它们提供可追溯性：对于生产中的任何模型，你应该能够追溯到它是如何训练的，用什么数据，由谁训练。通常称为数据和模型血缘
    - 它们促进协作：团队成员可以通过跟踪工具彼此分享结果，而不是通过电子邮件发送电子表格摘要

    如前一章所强调的，没有这些实践，ML团队面临很多痛苦：无法重现或信任的模型，因为你记不起哪个笔记本有那个伟大的结果而丢失的工作，合并多人贡献的困难，甚至部署灾难（部署错误的模型版本等）。

    ![没有版本控制的问题](https://www.dailydoseofds.com/content/images/2025/08/image-87.png)

    现在我们已经明确了案例，让我们看看W&B服务以及它们如何帮助解决这些问题。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Weights and Biases：核心理念

    W&B将自己定位为"开发者优先的MLOps平台"。它是基于云的，主要专注于实验跟踪、数据集/模型版本控制和协作。

    ![W&B平台](https://www.dailydoseofds.com/content/images/2025/08/image-88.png)

    > 👉 **声明**：我们与W&B没有任何关联。

    W&B的核心论点是，机器学习中最高杠杆的活动是训练模型、跟踪其性能、与以前的尝试进行比较，并决定下一步尝试什么的循环。

    W&B旨在使这个循环尽可能快速、有洞察力和协作。

    根据我们的经验，它以其交互式UI脱颖而出，提供仪表板来比较运行、可视化指标，甚至创建报告。

    虽然W&B主要作为托管服务提供，但它也支持自管理和本地部署。在本章中，我们的重点将完全是使用W&B作为托管服务。

    W&B与许多框架（PyTorch、TensorFlow、scikit-learn等）开箱即用集成，便于记录。

    在对Weights & Biases及其方法有了非常基本的了解之后，理解它与MLflow的区别也变得重要。让我们快速看一下这一点。

    ---

    ## 对比：MLflow vs. W&B

    | 特性/方面 | **MLflow** | **Weights & Biases (W&B)** |
    |-----------|------------|----------------------------|
    | **性质** | 开源，自托管（本地或服务器） | 云优先，托管（免费和付费层） |
    | **实验跟踪** | 记录参数、指标、工件 | 类似但具有更丰富的可视化 |
    | **UI** | 基本Web UI，简单图表 | 高级仪表板与交互式图表 |
    | **协作** | 有限 | 强大：团队仪表板、报告 |
    | **工件存储** | 本地（默认） | 托管（或与集成的外部存储桶） |
    | **易用性** | 简单Python API，更多手动配置 | 用户友好，大量集成（PyTorch、Keras、HuggingFace） |
    | **离线使用** | 完全可能（本地记录+UI） | 离线可能，但主要优势在线上 |
    | **最适合** | 本地/企业设置，自定义基础设施 | 快速设置，协作，可视化重度工作流 |

    因此，回答"如果我已经知道MLflow，为什么要学习W&B？"

    MLflow和W&B都是顶级的，但如果你或你的团队不想要设置和维护的麻烦，W&B是更好的选择，因为：

    - **托管vs自托管**：MLflow通常需要跟踪服务器；W&B可以提供完全托管的SaaS
    - **实验跟踪**：两者都记录运行/指标，但W&B开箱即用提供更丰富的可视化、仪表板和协作
    - **工件和注册表**：W&B无缝集成工件存储和模型注册表，而MLflow的更基本，除非在Databricks上
    - **协作**：W&B面向团队，具有易于分享和报告的功能；MLflow更像是一个灵活的工具包

    > 👉 **核心要点**：W&B减少了基础设施开销并提升了协作/可视化，而MLflow更精简但需要自管理。两者都不是天生优越的——正确的选择完全取决于用例和应用场景。

    有了这个温和的介绍和比较，让我们直接跳入实现，看看我们将如何在ML工作流中使用W&B。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 使用scikit-learn进行预测建模

    让我们使用Weights & Biases构建我们的第一个完整的、可重现的机器学习项目。

    我们将从经典的房价预测问题开始，这个问题非常适合演示核心W&B工作流，而不会在模型复杂性中迷失。

    这是一个基础的回归问题，一个可靠的模型可以为从房产估值到投资策略的一切提供动力。

    对于这个演练，我们将使用著名的加利福尼亚房价数据集，但这些原则适用于任何表格回归任务，例如预测客户生命周期价值、预测库存需求或估算保险理赔金额。

    我们的目标是使用scikit-learn训练一个`RandomForestRegressor`模型，并在此过程中使用W&B构建一个完全版本化和可重现的管道。我们将系统地：

    - 使用W&B Artifacts对我们的原始数据集进行版本控制
    - 跟踪我们的训练实验，记录超参数和评估指标
    - 利用W&B内置的scikit-learn集成
    - 将最终训练的模型作为工件进行版本控制
    - 将我们的最佳模型链接到W&B注册表，将其标记为暂存候选

    ### 项目设置

    这个项目的代码旨在使用Google Colab运行。我们建议在Colab中上传`.ipynb`笔记本，并从那里运行它。

    [下载笔记本](https://www.dailydoseofds.com/content/files/2025/08/demo-one.ipynb)

    ### 账户设置

    首先，我们需要在[W&B](https://wandb.ai/site)平台上注册。按照以下步骤设置免费账户：

    1. **注册账户**
       - 点击"Sign Up"按钮并填写你的详细信息
       - 你将收到一封验证身份的电子邮件来激活你的账户
       - 通过点击收到的链接进行验证

    2. **完成账户创建**
       - 在提示页面上输入所有必需的详细信息
       - 通过选择"Professional"账户完成账户创建
       - 如果你是学生/研究人员，你可能想选择"Academic"账户类型，因为W&B学术计划更加宽松

    3. **账户类型说明**
       - 如果你选择了"Professional"账户，你将获得"Pro"计划的30天试用
       - 试用期后，如果不支付月费，账户将降级到"Free"层，这对于学习和实验W&B的核心功能也相当慷慨

    现在我们的账户已经成功设置，让我们继续看看如何将我们的脚本/笔记本连接到W&B云。

    ### 将笔记本连接到W&B云

    为了使用W&B服务和跟踪任何内容，我们需要在笔记本和Weights & Biases平台之间建立连接。

    为此，我们需要使用登录命令从笔记本登录到我们的W&B账户：

    ```bash
    !wandb login
    ```

    > 💡 **提示**：如果你不知道，你可以通过在命令前加上"!"在Jupyter笔记本中运行shell命令。所以你需要在Colab上运行的笔记本中运行上述命令。

    运行此单元格后，登录命令将提示你粘贴API密钥，你可以在账户设置中找到该密钥，或使用单元格输出中提供的链接（以绿色突出显示）。

    > 👉 **注意**：如果你点击单元格输出中的链接，W&B要求你重新登录，那么只需这样做并返回单元格输出，再次点击链接以获取你的API密钥。

    提供你的API密钥并按Enter/Return键。这将登录到W&B上的注册账户，有效地将我们的笔记本连接到W&B平台。

    此密钥验证你的机器并告诉`wandb`库将数据发送到哪里。

    连接设置完成后，让我们继续看看我们将如何使用W&B在数据和模型代码中利用其功能。

    ### 使用W&B artifacts进行数据集版本控制

    构建可重现管道的第一步是对我们的数据集进行版本控制。如果我们不能保证使用完全相同的数据，我们就不能保证可重现的结果。

    在W&B中，对任何文件或文件集合进行版本控制的机制是W&B Artifacts。

    **将artifact想象为一个版本化的、云支持的文件夹。**

    让我们看看加载房屋数据、将其保存为CSV，然后将其作为我们的第一个artifact记录到W&B的代码：

    ```python
    import wandb
    import pandas as pd
    from sklearn.datasets import fetch_california_housing
    import numpy as np

    # 初始化W&B项目
    wandb.init(
        project="house-price-prediction",
        name="data-preparation",
        job_type="data-prep"
    )

    # 加载加利福尼亚房屋数据集
    housing = fetch_california_housing()

    # 创建DataFrame
    df = pd.DataFrame(
        housing.data,
        columns=housing.feature_names
    )
    df['target'] = housing.target

    # 保存为CSV文件
    df.to_csv('california_housing.csv', index=False)

    # 创建W&B artifact
    raw_data_artifact = wandb.Artifact(
        name="california-housing-raw",
        type="dataset",
        description="原始加利福尼亚房屋数据集"
    )

    # 添加文件到artifact
    raw_data_artifact.add_file('california_housing.csv')

    # 记录artifact到W&B
    wandb.log_artifact(raw_data_artifact)

    # 完成运行
    wandb.finish()
    ```

    这段代码做了几件重要的事情：

    1. **初始化W&B运行**：`wandb.init()`创建一个新的实验运行
    2. **创建artifact**：`wandb.Artifact()`定义一个新的版本化资产
    3. **添加文件**：`add_file()`将我们的CSV文件添加到artifact
    4. **记录artifact**：`log_artifact()`将其上传到W&B云

    一旦运行，你将在W&B仪表板中看到这个artifact，完整的版本历史和血缘信息。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 实验跟踪和模型训练

    现在我们有了版本化的数据集，让我们训练我们的模型并跟踪整个实验。这是W&B真正发光的地方——它可以自动记录超参数、指标，甚至模型工件。

    ```python
    import wandb
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib

    # 初始化W&B运行
    wandb.init(
        project="house-price-prediction",
        name="random-forest-training",
        job_type="train"
    )

    # 配置超参数
    config = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "test_size": 0.2
    }

    # 将配置记录到W&B
    wandb.config.update(config)

    # 从W&B下载数据集artifact
    artifact = wandb.use_artifact('california-housing-raw:latest')
    artifact_dir = artifact.download()

    # 加载数据
    df = pd.read_csv(f'{artifact_dir}/california_housing.csv')

    # 准备特征和目标
    X = df.drop('target', axis=1)
    y = df['target']

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['test_size'],
        random_state=config['random_state']
    )

    # 训练模型
    model = RandomForestRegressor(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        min_samples_split=config['min_samples_split'],
        min_samples_leaf=config['min_samples_leaf'],
        random_state=config['random_state']
    )

    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 计算指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # 记录指标到W&B
    wandb.log({
        "mse": mse,
        "rmse": rmse,
        "r2_score": r2,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    })

    # 保存模型
    model_filename = 'random_forest_model.joblib'
    joblib.dump(model, model_filename)

    # 创建模型artifact
    model_artifact = wandb.Artifact(
        name="random-forest-model",
        type="model",
        description=f"随机森林回归模型，RMSE: {rmse:.4f}"
    )

    model_artifact.add_file(model_filename)
    wandb.log_artifact(model_artifact)

    # 记录特征重要性
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # 创建特征重要性图表
    wandb.log({
        "feature_importance": wandb.plot.bar(
            wandb.Table(dataframe=feature_importance),
            "feature", "importance",
            title="特征重要性"
        )
    })

    print(f"模型训练完成！")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    wandb.finish()
    ```

    这个训练脚本展示了W&B的几个强大功能：

    #### 🔧 **配置管理**
    - `wandb.config.update()`自动记录所有超参数
    - 这些配置在W&B UI中可见，便于比较不同运行

    #### 📊 **指标记录**
    - `wandb.log()`记录训练和验证指标
    - 指标自动可视化在交互式图表中

    #### 🎯 **Artifact血缘**
    - `wandb.use_artifact()`创建数据血缘
    - W&B跟踪哪个数据版本用于训练哪个模型

    #### 📈 **可视化**
    - `wandb.plot.bar()`创建自定义可视化
    - 特征重要性等洞察直接嵌入到实验中

    ### 模型注册和版本管理

    训练完成后，我们可以将最佳模型提升到W&B模型注册表：

    ```python
    # 将模型链接到注册表
    wandb.init(
        project="house-price-prediction",
        job_type="model-registry"
    )

    # 获取最佳模型artifact
    model_artifact = wandb.use_artifact('random-forest-model:latest')

    # 将模型链接到注册表
    wandb.link_artifact(
        artifact=model_artifact,
        target_path="house-price-predictor",
        aliases=["staging", "candidate"]
    )

    print("模型已成功注册到模型注册表！")
    wandb.finish()
    ```

    ### W&B仪表板功能

    完成这些步骤后，你将在W&B仪表板中看到：

    #### 📊 **实验跟踪**
    - 所有运行的完整历史
    - 超参数和指标的并排比较
    - 交互式图表和可视化

    #### 🗂️ **Artifact管理**
    - 数据集和模型的版本历史
    - 完整的血缘图显示数据→模型→部署的流程
    - 每个artifact的元数据和描述

    #### 🎯 **模型注册表**
    - 生产就绪模型的中央存储库
    - 模型状态管理（开发→暂存→生产）
    - 模型性能比较和A/B测试支持

    #### 👥 **协作功能**
    - 团队成员可以查看和比较实验
    - 报告生成和分享
    - 评论和讨论功能
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## PyTorch时间序列预测示例

    让我们通过一个更复杂的例子来展示W&B在深度学习工作流中的强大功能。我们将构建一个LSTM模型来预测时间序列数据。

    ### 数据准备和版本控制

    ```python
    import wandb
    import torch
    import torch.nn as nn
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import matplotlib.pyplot as plt

    # 初始化W&B项目
    wandb.init(
        project="time-series-forecasting",
        name="data-preparation",
        job_type="data-prep"
    )

    # 生成合成时间序列数据
    np.random.seed(42)
    time_steps = 1000
    t = np.linspace(0, 100, time_steps)

    # 创建复杂的时间序列：趋势 + 季节性 + 噪声
    trend = 0.02 * t
    seasonal = 10 * np.sin(0.5 * t) + 5 * np.cos(0.3 * t)
    noise = np.random.normal(0, 2, time_steps)
    ts_data = trend + seasonal + noise + 50

    # 创建DataFrame
    df = pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=time_steps, freq='D'),
        'value': ts_data
    })

    # 保存数据
    df.to_csv('time_series_data.csv', index=False)

    # 创建数据可视化
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['value'])
    plt.title('时间序列数据')
    plt.xlabel('时间')
    plt.ylabel('值')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('time_series_plot.png', dpi=150, bbox_inches='tight')

    # 记录图表到W&B
    wandb.log({"时间序列数据": wandb.Image('time_series_plot.png')})

    # 创建数据artifact
    data_artifact = wandb.Artifact(
        name="time-series-data",
        type="dataset",
        description="合成时间序列数据用于LSTM训练"
    )

    data_artifact.add_file('time_series_data.csv')
    data_artifact.add_file('time_series_plot.png')
    wandb.log_artifact(data_artifact)

    wandb.finish()
    ```

    ### LSTM模型定义

    ```python
    class LSTMPredictor(nn.Module):
        def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
            super(LSTMPredictor, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            self.dropout = nn.Dropout(dropout)
            self.linear = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            # 初始化隐藏状态
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

            # LSTM前向传播
            out, _ = self.lstm(x, (h0, c0))

            # 只使用最后一个时间步的输出
            out = self.dropout(out[:, -1, :])
            out = self.linear(out)
            return out

    def create_sequences(data, seq_length):
        \"\"\"创建用于LSTM训练的序列\"\"\"
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    ```

    ### 模型训练与W&B集成

    ```python
    # 初始化训练运行
    wandb.init(
        project="time-series-forecasting",
        name="lstm-training",
        job_type="train"
    )

    # 超参数配置
    config = {
        "sequence_length": 30,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "train_split": 0.8
    }

    wandb.config.update(config)

    # 下载数据artifact
    artifact = wandb.use_artifact('time-series-data:latest')
    artifact_dir = artifact.download()

    # 加载和预处理数据
    df = pd.read_csv(f'{artifact_dir}/time_series_data.csv')
    data = df['value'].values.reshape(-1, 1)

    # 数据标准化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # 创建序列
    X, y = create_sequences(scaled_data, config['sequence_length'])

    # 分割训练和测试数据
    split_idx = int(len(X) * config['train_split'])
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )

    # 初始化模型
    model = LSTMPredictor(
        input_size=1,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # 训练循环
    model.train()
    for epoch in range(config['epochs']):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # 验证
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test).item()

                # 反标准化预测结果
                test_pred_scaled = test_outputs.numpy()
                test_true_scaled = y_test.numpy()

                test_pred = scaler.inverse_transform(test_pred_scaled)
                test_true = scaler.inverse_transform(test_true_scaled)

                mae = mean_absolute_error(test_true, test_pred)
                rmse = np.sqrt(mean_squared_error(test_true, test_pred))

            # 记录指标到W&B
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "test_loss": test_loss,
                "mae": mae,
                "rmse": rmse
            })

            model.train()

    # 保存模型
    torch.save(model.state_dict(), 'lstm_model.pth')

    # 创建模型artifact
    model_artifact = wandb.Artifact(
        name="lstm-time-series-model",
        type="model",
        description=f"LSTM时间序列预测模型，RMSE: {rmse:.4f}"
    )

    model_artifact.add_file('lstm_model.pth')
    wandb.log_artifact(model_artifact)

    print(f"训练完成！最终RMSE: {rmse:.4f}")
    wandb.finish()
    ```

    ### 预测可视化和模型评估

    ```python
    # 初始化评估运行
    wandb.init(
        project="time-series-forecasting",
        name="model-evaluation",
        job_type="eval"
    )

    # 加载训练好的模型
    model_artifact = wandb.use_artifact('lstm-time-series-model:latest')
    model_dir = model_artifact.download()

    # 重新创建模型并加载权重
    model = LSTMPredictor(hidden_size=64, num_layers=2, dropout=0.2)
    model.load_state_dict(torch.load(f'{model_dir}/lstm_model.pth'))
    model.eval()

    # 生成预测
    with torch.no_grad():
        predictions = model(X_test)

    # 反标准化
    pred_values = scaler.inverse_transform(predictions.numpy())
    true_values = scaler.inverse_transform(y_test.numpy())

    # 创建预测vs真实值的对比图
    plt.figure(figsize=(15, 8))

    # 绘制最后200个点的预测结果
    plot_range = slice(-200, None)
    plt.plot(true_values[plot_range], label='真实值', alpha=0.8)
    plt.plot(pred_values[plot_range], label='预测值', alpha=0.8)
    plt.title('LSTM时间序列预测结果')
    plt.xlabel('时间步')
    plt.ylabel('值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('prediction_comparison.png', dpi=150, bbox_inches='tight')

    # 记录预测结果图
    wandb.log({
        "预测对比": wandb.Image('prediction_comparison.png'),
        "最终MAE": mean_absolute_error(true_values, pred_values),
        "最终RMSE": np.sqrt(mean_squared_error(true_values, pred_values))
    })

    # 创建散点图显示预测准确性
    plt.figure(figsize=(8, 8))
    plt.scatter(true_values, pred_values, alpha=0.5)
    plt.plot([true_values.min(), true_values.max()],
            [true_values.min(), true_values.max()], 'r--', lw=2)
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('预测准确性散点图')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('accuracy_scatter.png', dpi=150, bbox_inches='tight')

    wandb.log({"准确性散点图": wandb.Image('accuracy_scatter.png')})

    wandb.finish()
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## W&B高级功能

    ### 1. 超参数扫描 (Hyperparameter Sweeps)

    W&B的扫描功能允许你自动化超参数调优过程：

    ```python
    # 定义扫描配置
    sweep_config = {
        'method': 'bayes',  # 贝叶斯优化
        'metric': {
            'name': 'rmse',
            'goal': 'minimize'
        },
        'parameters': {
            'hidden_size': {
                'values': [32, 64, 128, 256]
            },
            'num_layers': {
                'values': [1, 2, 3]
            },
            'learning_rate': {
                'distribution': 'log_uniform',
                'min': 1e-5,
                'max': 1e-2
            },
            'dropout': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.5
            }
        }
    }

    # 创建扫描
    sweep_id = wandb.sweep(sweep_config, project="time-series-forecasting")

    # 定义训练函数
    def train_with_sweep():
        wandb.init()
        config = wandb.config

        # 使用config中的超参数训练模型
        model = LSTMPredictor(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout
        )

        # ... 训练代码 ...

        # 记录最终指标
        wandb.log({"rmse": final_rmse})

    # 运行扫描
    wandb.agent(sweep_id, train_with_sweep, count=20)
    ```

    ### 2. 模型血缘和可追溯性

    W&B自动跟踪模型血缘，显示：
    - 哪个数据集用于训练哪个模型
    - 模型的完整训练历史
    - 从数据到部署的完整流程

    ```python
    # 查看artifact血缘
    artifact = wandb.use_artifact('lstm-time-series-model:latest')

    # 获取血缘信息
    lineage = artifact.logged_by()  # 哪个运行创建了这个artifact
    used_by = artifact.used_by()    # 哪些运行使用了这个artifact

    print(f"模型由运行创建: {lineage.name}")
    print(f"模型被以下运行使用: {[run.name for run in used_by]}")
    ```

    ### 3. 报告和协作

    W&B允许创建丰富的报告来分享实验结果：

    ```python
    # 在W&B UI中创建报告
    # 1. 转到你的项目页面
    # 2. 点击"Reports"标签
    # 3. 创建新报告
    # 4. 添加图表、表格、文本和图像
    # 5. 分享给团队成员
    ```

    ### 4. 模型注册表管理

    ```python
    # 将模型提升到不同阶段
    wandb.init(project="time-series-forecasting", job_type="model-promotion")

    # 获取最佳模型
    best_model = wandb.use_artifact('lstm-time-series-model:latest')

    # 提升到staging
    wandb.link_artifact(
        artifact=best_model,
        target_path="time-series-predictor",
        aliases=["staging", "v1.0"]
    )

    # 提升到production（在验证后）
    wandb.link_artifact(
        artifact=best_model,
        target_path="time-series-predictor",
        aliases=["production", "v1.0"]
    )

    wandb.finish()
    ```

    ### 5. 自定义指标和可视化

    ```python
    # 自定义表格
    columns = ["epoch", "train_loss", "val_loss", "learning_rate"]
    data = [[1, 0.5, 0.6, 0.001], [2, 0.4, 0.5, 0.001]]
    table = wandb.Table(data=data, columns=columns)
    wandb.log({"训练历史": table})

    # 自定义图表
    wandb.log({
        "损失对比": wandb.plot.line_series(
            xs=[1, 2, 3, 4, 5],
            ys=[[0.5, 0.4, 0.3, 0.25, 0.2], [0.6, 0.5, 0.4, 0.35, 0.3]],
            keys=["训练损失", "验证损失"],
            title="训练过程中的损失变化",
            xname="Epoch"
        )
    })

    # 3D散点图
    wandb.log({
        "3D预测": wandb.Object3D({
            "type": "lidar/beta",
            "points": np.random.rand(100, 3),
            "colors": np.random.rand(100, 3)
        })
    })
    ```

    ### 6. 离线模式和同步

    ```python
    # 离线模式
    import os
    os.environ["WANDB_MODE"] = "offline"

    wandb.init(project="offline-project")
    # ... 训练代码 ...
    wandb.finish()

    # 稍后同步
    # wandb sync wandb/offline-run-xxx
    ```

    ### W&B vs 其他工具的优势总结

    | 功能 | W&B | MLflow | TensorBoard |
    |------|-----|--------|-------------|
    | **设置复杂度** | 极简 | 中等 | 简单 |
    | **协作功能** | 优秀 | 基础 | 无 |
    | **可视化** | 丰富交互式 | 基础 | 专业但有限 |
    | **超参数扫描** | 内置高级 | 基础 | 无 |
    | **模型注册表** | 完整 | 完整 | 无 |
    | **报告生成** | 优秀 | 无 | 无 |
    | **云托管** | 是 | 可选 | 无 |
    | **成本** | 免费层慷慨 | 开源 | 免费 |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## W&B最佳实践

    ### 1. 项目组织策略

    #### 🏗️ **项目结构建议**
    ```
    项目命名约定：
    - 使用描述性名称：house-price-prediction
    - 避免空格，使用连字符
    - 包含团队或部门前缀：ml-team-house-prediction

    运行命名约定：
    - 包含实验类型：baseline-rf, optimized-lstm
    - 添加日期或版本：v1.0-baseline, 2024-01-15-experiment
    - 使用有意义的描述：feature-engineering-v2
    ```

    #### 📊 **Artifact组织**
    ```python
    # 数据artifact命名
    raw_data = wandb.Artifact("raw-data-v1.0", type="dataset")
    processed_data = wandb.Artifact("processed-data-v1.0", type="dataset")

    # 模型artifact命名
    model = wandb.Artifact("model-v1.0", type="model")

    # 使用语义化版本控制
    # v1.0.0 - 主要版本.次要版本.补丁版本
    ```

    ### 2. 实验跟踪最佳实践

    #### 🔧 **配置管理**
    ```python
    # 使用配置字典统一管理超参数
    config = {
        # 模型参数
        "model": {
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2
        },
        # 训练参数
        "training": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100
        },
        # 数据参数
        "data": {
            "sequence_length": 30,
            "train_split": 0.8
        }
    }

    wandb.config.update(config)
    ```

    #### 📈 **指标记录策略**
    ```python
    # 记录多种指标类型
    wandb.log({
        # 损失指标
        "train_loss": train_loss,
        "val_loss": val_loss,

        # 业务指标
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,

        # 系统指标
        "gpu_memory": gpu_memory_usage,
        "training_time": epoch_time,

        # 自定义指标
        "custom_metric": custom_value
    })
    ```

    ### 3. 团队协作工作流

    #### 👥 **团队项目设置**
    ```python
    # 使用团队workspace
    wandb.init(
        entity="your-team-name",  # 团队名称
        project="shared-project",
        name="experiment-name",
        tags=["baseline", "team-member-name"]  # 使用标签分类
    )
    ```

    #### 🔄 **代码审查集成**
    ```python
    # 在代码中包含W&B链接
    def train_model():
        run = wandb.init(project="my-project")

        # 训练代码...

        print(f"实验结果: {run.url}")
        return run.url

    # 在PR描述中包含实验链接
    # 便于代码审查时查看实验结果
    ```

    ### 4. 生产部署集成

    #### 🚀 **模型部署工作流**
    ```python
    # 1. 训练阶段
    wandb.init(project="production-model", job_type="train")
    # ... 训练代码 ...
    model_artifact = wandb.Artifact("model", type="model")
    wandb.log_artifact(model_artifact)

    # 2. 验证阶段
    wandb.init(project="production-model", job_type="validate")
    model = wandb.use_artifact("model:latest")
    # ... 验证代码 ...
    if validation_passed:
        wandb.link_artifact(model, "model-registry", aliases=["staging"])

    # 3. 部署阶段
    wandb.init(project="production-model", job_type="deploy")
    staging_model = wandb.use_artifact("model-registry:staging")
    # ... 部署代码 ...
    wandb.link_artifact(staging_model, "model-registry", aliases=["production"])
    ```

    #### 📊 **生产监控**
    ```python
    # 在生产环境中记录模型性能
    wandb.init(project="production-monitoring", job_type="inference")

    # 记录推理指标
    wandb.log({
        "inference_latency": latency,
        "prediction_confidence": confidence,
        "data_drift_score": drift_score,
        "model_accuracy": accuracy
    })
    ```

    ### 5. 成本优化策略

    #### 💰 **存储优化**
    ```python
    # 使用外部存储减少W&B存储成本
    artifact = wandb.Artifact("large-dataset", type="dataset")

    # 引用外部文件而不是上传
    artifact.add_reference("s3://my-bucket/large-file.csv")

    # 或使用压缩
    artifact.add_file("data.csv.gz")
    ```

    #### ⚡ **记录优化**
    ```python
    # 批量记录减少API调用
    metrics_batch = {}
    for epoch in range(num_epochs):
        # ... 训练 ...
        metrics_batch[f"epoch_{epoch}_loss"] = loss

        # 每10个epoch记录一次
        if epoch % 10 == 0:
            wandb.log(metrics_batch)
            metrics_batch = {}
    ```

    ### 6. 安全和隐私考虑

    #### 🔒 **敏感数据处理**
    ```python
    # 不要记录敏感信息
    config = {
        "model_params": {...},
        # 不要包含API密钥、密码等
        # "api_key": "secret_key"  # ❌ 错误
    }

    # 使用环境变量
    import os
    api_key = os.getenv("API_KEY")  # ✅ 正确
    ```

    #### 🛡️ **访问控制**
    ```python
    # 使用私有项目处理敏感数据
    wandb.init(
        project="private-project",
        entity="your-team",
        mode="online",  # 确保数据加密传输
    )
    ```

    ### 7. 调试和故障排除

    #### 🐛 **常见问题解决**
    ```python
    # 1. 网络问题 - 使用离线模式
    os.environ["WANDB_MODE"] = "offline"

    # 2. 大文件上传 - 使用引用
    artifact.add_reference("file://large_file.bin")

    # 3. 内存问题 - 分批记录
    for batch in data_batches:
        wandb.log({"batch_metric": process_batch(batch)})

    # 4. 调试模式
    wandb.init(project="debug", mode="disabled")  # 禁用记录
    ```

    #### 📝 **日志记录**
    ```python
    import logging

    # 设置W&B日志级别
    logging.getLogger("wandb").setLevel(logging.WARNING)

    # 记录自定义日志
    wandb.init(project="my-project")
    wandb.log({"custom_log": "Training started"})
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 总结与展望

    ### 🎯 **关键收获**

    通过这个深入的W&B实践指南，我们学到了：

    #### 📚 **核心概念掌握**
    - **可重现性的重要性**：在ML系统中确保实验可重复的关键价值
    - **版本控制策略**：数据、模型和实验的系统化版本管理
    - **血缘追踪**：从数据到模型到部署的完整可追溯性

    #### 🛠️ **实践技能获得**
    - **W&B平台使用**：从基础设置到高级功能的全面掌握
    - **实验跟踪**：系统化记录和比较ML实验
    - **Artifact管理**：数据集和模型的版本化存储和管理
    - **协作工作流**：团队环境中的ML项目协作

    #### 🔧 **技术实现能力**
    - **scikit-learn集成**：传统ML工作流的W&B集成
    - **PyTorch深度学习**：复杂神经网络训练的跟踪和管理
    - **超参数优化**：自动化调参和实验扫描
    - **生产部署**：从实验到生产的完整流程

    ### 🚀 **W&B的核心价值**

    #### ✅ **解决的核心问题**
    1. **实验混乱**：将无序的实验转化为有组织的知识积累
    2. **协作困难**：通过云平台实现无缝团队协作
    3. **结果不可重现**：通过系统化版本控制确保可重现性
    4. **模型血缘缺失**：提供完整的数据→模型→部署追踪

    #### 🎪 **独特优势**
    - **零配置启动**：相比MLflow的复杂设置，W&B即开即用
    - **丰富可视化**：交互式图表和仪表板超越传统工具
    - **智能协作**：内置团队功能和报告生成
    - **云原生设计**：无需维护基础设施的托管服务

    ### 🔍 **与其他工具的定位**

    #### 📊 **工具选择指南**

    **选择W&B的场景：**
    - 团队协作需求强烈
    - 希望快速启动，避免基础设施维护
    - 重视可视化和交互式分析
    - 需要丰富的实验比较功能

    **选择MLflow的场景：**
    - 需要完全控制基础设施
    - 有严格的数据本地化要求
    - 预算有限，偏好开源解决方案
    - 已有成熟的ML基础设施

    **选择DVC的场景：**
    - 主要需求是数据版本控制
    - 与Git工作流深度集成
    - 偏好轻量级解决方案
    - 数据存储在本地或自有云存储

    ### 🎯 **最佳实践总结**

    #### 🏗️ **项目组织**
    - 使用清晰的命名约定
    - 合理的项目和实验分组
    - 标签和描述的有效使用

    #### 📈 **实验管理**
    - 系统化的超参数记录
    - 多维度指标跟踪
    - 定期的实验清理和归档

    #### 👥 **团队协作**
    - 统一的工作流程
    - 清晰的角色和权限管理
    - 有效的知识分享机制

    #### 🚀 **生产集成**
    - 模型注册表的规范使用
    - 部署流程的自动化
    - 生产监控的持续跟踪

    ### 🔮 **未来发展方向**

    #### 🤖 **技术趋势**
    - **AutoML集成**：自动化机器学习流程的深度集成
    - **边缘部署**：支持边缘设备的模型部署和监控
    - **联邦学习**：分布式学习场景的实验跟踪
    - **大模型支持**：LLM训练和微调的专门优化

    #### 🌐 **生态系统扩展**
    - **更多框架集成**：支持新兴ML框架
    - **云平台集成**：与主要云服务商的深度集成
    - **企业功能**：更强的安全性和合规性支持
    - **开源贡献**：社区驱动的功能开发

    ### 💡 **实践建议**

    #### 🎯 **立即行动**
    1. **创建W&B账户**：开始你的第一个实验
    2. **迁移现有项目**：将当前项目逐步迁移到W&B
    3. **建立团队规范**：制定团队的W&B使用标准
    4. **持续学习**：关注W&B的新功能和最佳实践

    #### 📚 **深入学习**
    - 探索W&B官方文档和教程
    - 参与社区讨论和案例分享
    - 实践不同类型的ML项目
    - 关注MLOps领域的最新发展

    ### 🎪 **结语**

    我们看到了W&B的实验跟踪、数据集/模型版本控制、血缘图和注册表功能如何为快速发展的ML项目带来秩序、可追溯性和可重复性。

    在此过程中，我们构建了两个完全可重现的工作流：一个在scikit-learn中用于表格回归，另一个在PyTorch中用于时间序列预测，每个都展示了端到端的artifact管理和模型提升。

    **关键要点是，可重现性不是"锦上添花"，而是成熟ML系统的结构性属性。**

    有了正确的工具，它变成了第二天性：协作的推动者、更快迭代的促进者、更安全部署的保障者，以及长期可维护性的基础。

    随着我们展望这个系列的未来章节，我们将探索：

    - **数据处理和管道**
    - **为ML系统量身定制的CI/CD工作流**
    - **来自行业的真实案例研究**
    - **模型开发和实践**
    - **生产中的监控和观察**
    - **LLMOps的特殊考虑**
    - **结合生命周期所有元素的完整端到端示例**

    请注意，到目前为止，我们主要看到了DVC、MLflow和W&B；然而，实现细节可能因用例、规模和行业/公司而异。

    因此，深入理解底层系统设计和生命周期原则至关重要。掌握核心知识将使你能够很好地驾驭任何MLOps堆栈或适应任何LLMOps场景。

    所以，随着我们前进，期待看到理论、方法和模拟的持续融合，这些将弥合实验和生产之间的差距。

    **目标，一如既往，是帮助你培养成熟的、以系统为中心的思维方式，将机器学习不视为独立的工件，而是更广泛软件生态系统的活跃部分。**

    ---

    🚀 **开始你的W&B之旅，让可重现性成为你ML工作流的核心！**
    """
    )
    return


if __name__ == "__main__":
    app.run()
