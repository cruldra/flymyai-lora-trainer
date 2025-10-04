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
    # MLOps完整蓝图：模型开发与优化（Part A）

    ## 📚 引言

    在生产环境中（例如Web服务或移动应用），推理延迟、吞吐量、内存占用和可扩展性等因素与原始预测性能同样重要。

    ![生产环境考虑因素](https://www.dailydoseofds.com/content/images/2025/09/image-16.png)

    ### 生产环境的现实

    一个准确率高1%但速度慢一倍、或者对于部署环境来说太大或太复杂的模型，在实践中可能是一个糟糕的选择。

    **这种关注点的转变是MLOps的核心**：我们需要技术来开发、优化和微调模型，使其不仅准确，而且在生产中可靠、快速、简单且具有成本效益。

    ![MLOps核心关注点](https://www.dailydoseofds.com/content/images/2025/09/image-17.png)

    ### 🎬 Netflix Prize的教训

    一个著名的真实案例来自Netflix Prize竞赛：

    - **获胜方案**：多个算法的集成，准确率提升10%
    - **实际部署**：Netflix从未部署它
    - **原因**：太复杂，难以大规模运行
    - **最终选择**：更简单、更易维护、更快的方案

    ![Netflix Prize案例](https://www.dailydoseofds.com/content/images/2025/09/Screenshot-2025-09-10-232914-1.png)

    ![复杂度与实用性的平衡](https://www.dailydoseofds.com/content/images/2025/09/image-18-1.png)

    **关键启示**：作为ML工程师，我们必须在模型复杂度和实用性之间取得平衡。

    ### 本章目标

    探讨如何选择和构建模型，使其既符合ML目标，又满足生产要求。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🎯 模型开发基础

    开发ML模型是一个迭代过程：选择方法 → 训练/评估 → 识别改进 → 重复，直到模型足够好可以部署。

    ![模型开发迭代过程](https://www.dailydoseofds.com/content/images/2025/09/image-16.png)

    ### "足够好"的定义

    在MLOps环境中，"足够好"意味着：

    - ✅ 达到可接受的准确率或误差指标
    - ✅ 满足速度要求
    - ✅ 满足内存限制
    - ✅ 具有可解释性（如果需要）
    - ✅ 满足其他约束条件

    ### 本节内容

    - 如何为问题选择正确的模型
    - 如何从简单到复杂逐步推进
    - 需要考虑哪些权衡
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 💡 模型选择和启动技巧

    选择正确的模型或算法是关键的第一步。很容易被最新的SOTA（state-of-the-art）技术所吸引，但"最佳"模型取决于上下文：数据大小、延迟需求、开发资源等。

    ### 1. 避免"最先进"陷阱 ⚠️

    **不要假设最新或最复杂的模型就是最佳解决方案。**

    - 前沿研究模型通常在学术基准上显示边际收益
    - 代价是复杂度大幅增加
    - 可能很慢、需要大量数据、难以实现

    ![SOTA陷阱](https://www.dailydoseofds.com/content/images/2025/09/image-19.png)

    **关键问题**：我真的需要一个十亿参数的Transformer吗？还是更简单的方法就足够了？

    **实践建议**：
    - 经过验证的方法通常更容易部署
    - 对于手头的任务足够有效
    - 谨慎使用SOTA模型
    - 评估其好处是否真正证明了生产环境中增加的复杂性

    ### 2. 从最简单的模型开始 🚀

    **指导原则**：简单胜于复杂。

    **为什么从简单开始？**

    - 更容易调试和部署
    - 快速验证管道
    - 提供基准

    **示例**：
    - 线性回归
    - 小型决策树
    - 逻辑回归

    **好处**：
    - 如果简单模型表现合理 → 确认特征包含信号
    - 提供基准 → 更复杂的模型应该超越它
    - 从简单开始，逐步增加复杂度 → 理解每个变化的影响

    **早期问题检测**：
    - 如果简单模型表现远低于预期 → 数据问题
    - 如果表现异常好 → 管道缺陷或数据泄漏

    ### 3. 避免模型比较中的偏见 ⚖️

    ![避免偏见](https://www.dailydoseofds.com/content/images/2025/09/image-20-1-1.png)

    **问题**：容易在你最感兴趣的模型上花更多时间调优，导致评估有偏。

    **解决方案**：
    - 给每个模型相同的关注和调优努力
    - 做出客观的、数据驱动的决策
    - 警惕人为偏见

    **确保公平比较**：
    - 使用相同的训练/验证分割
    - 使用相同的评估指标
    - 对每种模型类型进行足够的试验
    - 然后得出哪个模型族最有效的结论

    ### 4. 考虑当前与未来性能 📈

    **今天最好的模型可能不是明天最好的。**

    ![当前vs未来性能](https://www.dailydoseofds.com/content/images/2025/09/image-21.png)

    **数据增长的影响**：
    - 某些算法随着更多数据扩展得更好
    - 小数据集可能偏好决策树或SVM
    - 100倍数据后，神经网络可能在准确率上超越

    **实践建议**：
    - 绘制学习曲线：模型性能 vs 训练集大小
    - 如果一个模型的曲线快速平稳 → 数据增长帮助有限
    - 如果另一个模型的性能随数据持续改进 → 长期可能获胜

    ![学习曲线示例](https://www.dailydoseofds.com/content/images/2025/09/image-31.png)

    **适应性考虑**：
    - 模型需要频繁更新吗？
    - 可以增量学习的模型（在线学习）可能更好
    - 训练更快的模型可能更好
    - 即使即时准确率略低

    **真实案例**：
    - 协同过滤推荐器离线表现优于神经网络
    - 但神经网络可以在生产中从新数据实时学习
    - 部署后快速超越静态协同过滤器

    ### 5. 评估权衡 🔄

    不同的模型有不同的优缺点，你经常必须平衡权衡。

    ![权衡评估](https://www.dailydoseofds.com/content/images/2025/09/image-22.png)

    **常见权衡**：

    #### 准确率 vs 延迟
    - 更复杂的模型（深度神经网络、大型集成）通常产生更高的准确率
    - 但推理速度更慢
    - 实时欺诈检测：几毫秒内运行的略低准确率模型可能优于需要几秒的重型模型

    #### 准确率 vs 内存大小
    - 某些模型在内存中可能很大
    - 部署到内存受限环境（移动应用、IoT设备）需要更小的模型
    - 后续将讨论模型压缩方法

    #### 准确率 vs 可解释性
    - 简单模型（线性模型、小树）通常比复杂模型更可解释
    - 如果需要向用户或监管机构解释预测 → 选择更可解释的模型
    - 即使准确率有所损失

    #### 精确率 vs 召回率
    - 某些模型可以调整以强调其中一个
    - 取决于应用
    - 医疗筛查：假阴性（漏诊）比假阳性更糟 → 偏好捕获尽可能多阳性的模型

    #### 理解模型假设
    - 每个模型对数据做出隐式或显式假设
    - 线性回归：假设大致线性关系
    - 朴素贝叶斯：假设特征独立
    - 神经网络：假设数据是IID（独立同分布）
    - 如果这些假设严重违反 → 模型可能表现不佳
    - 将模型与问题匹配
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🔄 模型开发和部署的四个阶段

    成功的策略是分阶段迭代，从最简单到更复杂的解决方案。ML系统通常通过不同的成熟阶段演进。

    ### 阶段1：ML之前（使用启发式或简单规则） 📋

    **策略**：如果是第一次解决问题，尝试非ML基线。

    **示例**：
    - 电影推荐器：向所有人推荐前10部热门电影
    - 令人惊讶的是，这样的启发式通常提供合理的起点

    **Martin Zinkevich的ML规则**：
    - 如果你认为ML可以提供100%的提升
    - 简单的启发式可能提供50%

    **好处**：
    - 极快实现
    - 设定要超越的底线性能
    - 如果复杂模型无法超越朴素基线 → ML没有增加价值或有bug

    ![阶段1概念图](https://www.dailydoseofds.com/content/images/2025/09/image-33.png)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 阶段2：最简单的ML模型 🎯

    **时机**：启发式基线就位后（或启发式明显不足）

    **策略**：开发简单的ML模型

    **推荐算法**：
    - 逻辑回归
    - 基本决策树
    - K近邻

    **目标**：端到端验证ML管道

    **验证问题**：
    - 我们能在历史数据上训练并获得合理的预测吗？
    - 特征是否有信息量？
    - 模型在验证数据上的泛化能力是否优于启发式？

    **阶段2的价值**：
    - 证明ML对问题是可行的
    - 产生初始可部署模型
    - 简单模型更容易集成和服务

    **早期部署**：
    - 可以考虑部署以收集反馈
    - 逻辑回归可以快速投入生产
    - 开始积累性能数据或用户反馈
    - 同时研究更复杂的模型

    **MLOps实践**：
    - 鼓励早期部署简单模型
    - 在大量投资调优复杂模型之前测试所有管道
    - 数据收集、监控等

    ![阶段2概念图](https://www.dailydoseofds.com/content/images/2025/09/image-36.png)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 阶段3：优化简单模型 🔧

    **时机**：基本模型工作后

    **策略**：在不根本改变算法的情况下改进

    **优化方法**：

    #### 1. 特征工程
    - 创建和修改特征
    - 特征组合
    - 特征转换

    #### 2. 超参数调优
    - 系统地搜索更好的超参数
    - 后续章节将详细介绍策略

    #### 3. 目标函数调整
    - 优化不同的指标
    - 添加惩罚以解决特定目标
    - 类别不平衡、公平性等

    #### 4. 简单模型的集成
    - 组合多个模型
    - 训练多个决策树并平均（随机森林）
    - 组合逻辑回归和小型神经网络
    - 集成通常提升性能（通过平均减少方差）
    - Kaggle竞赛经常被集成赢得
    - 生产中较少见（增加复杂性）
    - 但适度的集成（几个不同的模型）仍然可管理且有益

    #### 5. 更多数据
    - 增强训练数据集
    - 如果可能收集更多样本
    - 向同一模型提供更多数据可以提高性能
    - 特别是如果模型数据饥渴（学习曲线可以告知）

    **阶段3的价值**：
    - 通常获得高投资回报
    - 利用更简单的模型（易于训练和理解）
    - 从中榨取所有性能

    **停止点**：
    - 许多应用ML系统可以在这里停止
    - 经过充分工程的逻辑回归或梯度提升树
    - 经过足够的调优和数据更新后可能满足所有要求

    ![阶段3概念图](https://www.dailydoseofds.com/content/images/2025/09/image-37.png)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 阶段4：复杂模型 🚀

    **时机**：只有在穷尽更简单的方法后（如果需要）

    **策略**：转向根本上更复杂的模型

    **可能的方向**：
    - 深度神经网络
    - Transformer架构
    - 取决于问题领域

    **要求**：
    - 通常需要更多数据
    - 更多计算资源
    - 可能引入新的工程挑战
      - 分布式训练
      - 专用硬件
      - 更长的推理时间

    **决策标准**：
    - 明确证据表明需要进一步改进
    - 无法通过改进阶段3解决方案获得
    - 例如：
      - 即使经过广泛调优，最佳模型的准确率仍然不足
      - 错误分析表明模型缺乏学习某些模式的能力

    **阶段4实践**：
    - 实验架构
    - 添加额外的网络层
    - 尝试预训练模型或迁移学习
    - 继续应用阶段3的经验教训
    - 继续应用特征工程、调优和集成

    **与MLOps的对齐**：
    - 到阶段4，通常已有完整的基础设施
    - 可能已部署更简单的模型或至少模拟了部署
    - 知道数据如何流动
    - 知道评估如何工作
    - 知道如何监控性能
    - 集成复杂模型更容易

    ![阶段4概念图](https://www.dailydoseofds.com/content/images/2025/09/image-38.png)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🔄 迭代和使用基线

    ### 核心原则

    **在每个阶段，使用前一阶段的最佳模型作为基线来衡量改进。**

    ### 权衡评估

    **场景**：阶段4的深度模型只略微改进指标，但速度明显更慢

    **考虑**：权衡是否值得？

    **替代方案**：
    - 阶段3优化的简单模型
    - 配合一些巧妙的压缩或集成
    - 可能在不需要极其复杂的解决方案的情况下达到目标

    ![四阶段总结图](https://www.dailydoseofds.com/content/images/2025/09/image-24.png)

    ### 关键要点

    - ✅ 从简单开始
    - ✅ 逐步增加复杂度
    - ✅ 每个阶段都有明确的目标
    - ✅ 使用前一阶段作为基线
    - ✅ 只在必要时增加复杂度
    - ✅ 权衡性能与复杂度
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---

    ## 🐛 调试模型训练

    即使有扎实的实验流程，你也会不可避免地遇到模型训练不佳或产生奇怪输出的情况。ML中的有效调试本身就是一项技能。

    ### ML调试的挑战

    与传统软件不同：
    - 没有确定性代码可以逐步执行
    - 涉及随机过程
    - 大型数据集
    - 可能以微妙方式失败的架构

    ![ML调试挑战](https://www.dailydoseofds.com/content/images/2025/09/image-45-1.png)

    ### 常见失败模式和调试技巧

    #### 1. 实现中的Bug 🐞

    **问题**：
    - 模型代码可能有错误
    - 损失计算错误
    - 张量操作不符合预期

    **解决方案**：
    - 对模型的部分进行单元测试
    - 使用小示例（可以手动计算预期输出）
    - 如果有参考实现（文献或其他框架）
    - 比较每层的输出以找出差异点

    #### 2. 超参数选择不当 ⚙️

    **问题**：
    - 模型可能完全适合任务
    - 但由于糟糕的超参数无法收敛或欠拟合

    **解决方案**：
    - 系统调优
    - 使用类似问题的已知良好默认值
    - 后续将详细讨论超参数调优

    #### 3. 数据问题 📊

    **原则**：垃圾进，垃圾出

    **问题**：
    - 如果训练数据有缺陷，模型也会有缺陷

    **解决方案**：
    - 始终对数据进行健全性检查
    - 基本统计
    - 可视化一些示例
    - 确保目标不是与ID或时间戳或其他伪影简单相关
    - 如果模型性能异常好或坏，重新审视数据预处理管道

    ![数据问题诊断](https://www.dailydoseofds.com/content/images/2025/09/image-44.png)

    #### 4. 特征问题 🎯

    **问题**：
    - 太多特征 → 过拟合或训练缓慢
    - 太少或错误的特征 → 限制性能
    - 生产中动态不可用的特征（数据泄漏）

    **解决方案**：
    - 检查特征重要性（对于允许的模型）
    - 尝试消融（移除特征并查看性能是否下降）
    - 识别哪些特征真正重要
    - 警惕只在预测事件后才存在的特征

    ### 诊断：高偏差 vs 高方差

    **高偏差（欠拟合）**：
    - 需要更复杂的模型
    - 需要更好的特征

    **高方差（过拟合）**：
    - 需要更多正则化
    - 需要更多数据
    - 需要更简单的模型

    ![偏差方差诊断](https://www.dailydoseofds.com/content/images/2025/09/image-42.png)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 经过验证的调试技术

    这些技术由经验丰富的从业者（包括Andrej Karpathy）倡导：

    #### 1. 再次从简单开始 🎯
    - 从最简单的版本开始
    - 对模型进行单元测试
    - 确保基本功能正常

    #### 2. 逐步增加复杂度 📈
    - 一次添加一个组件
    - 每次添加后验证
    - 更容易隔离问题

    #### 3. 使用一致的随机种子 🎲
    - 固定随机种子以确保可重现性
    - 消除随机性作为变量
    - 更容易比较运行

    #### 4. 密切监控训练指标 📊
    - 观察损失曲线
    - 检查训练和验证指标
    - 寻找异常模式

    #### 5. 检查中间输出和梯度（对于神经网络） 🔍
    - 验证激活值
    - 检查梯度大小
    - 识别梯度消失/爆炸

    #### 6. 使用调试工具和框架 🛠️
    - TensorBoard
    - Weights & Biases
    - 特定框架的调试器
    - 根据不同用例选择

    ![调试技术总结](https://www.dailydoseofds.com/content/images/2025/09/image-41.png)

    ### 业务逻辑和目标对齐

    **重要考虑**：
    - 有时模型在你设置的指标上表现良好
    - 但指标本身可能没有捕捉业务目标

    **示例**：
    - 你优化了准确率
    - 但真正需要的是某个类别的高召回率
    - 这可能不是模型的bug
    - 而是开发中的不对齐

    **验证**：
    - 始终验证离线评估与生产中关心的内容对齐
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 总结：ML模型调试

    调试ML模型涉及：

    ### 软件调试
    - 修复代码问题
    - 单元测试
    - 验证实现

    ### 科学调试
    - 诊断为什么学习算法没有达到预期结果
    - 理解模型行为
    - 调整方法

    ### 关键实践
    - 系统地跟踪实验
    - 应用这些调试策略
    - 以可靠的方式迭代模型开发

    现在我们将介绍优化技术（假设我们的模型训练正确，并且我们有跟踪实验的流程）。

    优化涉及理解"我们如何从模型中获得最大收益？"这包括调整超参数、微调预训练模型等。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---

    ## 🎛️ 模型优化

    在建立基线模型并确保训练管道健全后，下一个大的收益通常来自优化：找到最佳超参数并通过微调利用外部知识（如预训练模型）。

    ### 什么是优化？

    这些步骤可以被视为自动机器学习（AutoML）的形式：
    - 自动搜索更好的模型配置
    - 系统地探索超参数空间
    - 利用预训练知识

    ### 本节重点

    我们将深入讨论和理解**超参数优化（HPO）**。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🔧 超参数优化（HPO）

    ### 什么是超参数？

    超参数是在训练前设置的旋钮，例如：
    - 学习率
    - 层数
    - 树深度
    - 正则化强度
    - 批量大小
    - 等等

    ![超参数示例](https://www.dailydoseofds.com/content/images/2025/09/image-29.png)

    ### 为什么HPO重要？

    **影响巨大**：
    - 可以显著影响性能
    - 糟糕的超参数选择可能使强大的模型表现糟糕
    - 良好的选择可以产生SOTA结果

    **示例：神经网络学习率**
    - 太高 → 训练发散
    - 太低 → 训练太慢或陷入局部最小值

    **示例：随机森林**
    - 树不够或深度不够 → 欠拟合
    - 太多 → 过拟合或浪费计算

    ![HPO重要性](https://www.dailydoseofds.com/content/images/2025/09/image-27.png)

    **关键洞察**：
    - 调优良好的超参数可以让"较弱"的模型击败具有糟糕超参数的"较强"模型
    - 投资时间进行超参数调优通常会带来高回报

    ### 超参数 vs 模型参数

    **模型参数**：
    - 在训练期间学习
    - 例如：神经网络权重、线性回归系数

    **超参数**：
    - 在训练前选择
    - 通过实验和调优选择
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 手动 vs 自动调优

    #### 手动调优

    **早期阶段**：
    - 可能手动尝试几个值
    - 特别是如果你有直觉
    - 例如："也许较小的学习率会稳定训练"

    **缺点**：
    - 繁琐
    - 可能错过最优值
    - 特别是当超参数以非直观方式交互时

    #### 自动调优

    **更高效**：使用算法搜索方法

    ### 常见搜索方法

    #### 1. 网格搜索（Grid Search） 📊

    **方法**：
    - 为每个超参数定义离散值网格
    - 为每个组合训练模型

    **优点**：
    - 穷举搜索
    - 保证找到网格中的最佳组合

    **缺点**：
    - 组合爆炸
    - 如果有许多超参数，计算成本高

    **适用场景**：
    - 维度低
    - 对合理范围有粗略了解

    #### 2. 随机搜索（Random Search） 🎲

    **方法**：
    - 从分布中随机采样超参数
    - 而不是每个组合

    **优点**：
    - 研究表明比网格搜索更有效（预算有限时）
    - 特别是当只有少数超参数真正重要时
    - 在高维空间中探索更多组合

    **实践**：
    - 运行50个随机配置
    - 通常有很好的机会找到接近最优的区域

    ![搜索方法对比](https://www.dailydoseofds.com/content/images/2025/09/image-30.png)

    #### 3. 贝叶斯优化（Bayesian Optimization） 🧠

    **方法**：
    - 高斯过程优化
    - 树结构Parzen估计器
    - 基于过去结果建模超参数空间
    - 智能地猜测有希望的新点

    **优点**：
    - 可以在更少的运行中找到最优值
    - 智能引导搜索

    **缺点**：
    - 实现更复杂

    #### 4. 进化/遗传算法 🧬

    **方法**：
    - 将超参数集视为个体
    - 进化种群（突变/交叉）
    - 优化指标

    #### 5. 网格/随机 + 后续缩放 🔍

    **实用方法**：
    - 在广泛范围内进行粗略搜索（随机或网格）
    - 识别有希望的超参数区域
    - 在该区域进行更精细的搜索（放大）
    - 许多从业者使用这种两阶段方法

    ### 并行化

    **现代工具和云平台**：
    - 允许并行运行许多实验进行调优
    - 如果有计算资源，可以启动多个具有不同设置的训练作业
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### HPO实践技巧 💡

    #### 1. 选择合理的范围 📏

    **在启动搜索之前**：
    - 做一些研究
    - 使用先验知识
    - 为每个超参数设置范围

    **示例**：
    - 学习率：通常在 [1e-5, 1e-1]
    - 树深度：通常在 [3, 20]
    - 批量大小：通常是2的幂次

    #### 2. 评估指标 📊

    **使用验证集或交叉验证**：
    - 评估每个超参数设置
    - 不是测试集！

    **重要**：
    - 永远不要直接在测试集上调优
    - 那会泄漏测试信息
    - 你会过拟合到测试分割

    #### 3. 试验次数 🔢

    **快速训练（分钟或更少）**：
    - 可以负担数百次试验

    **慢速训练（小时）**：
    - 受限制

    **策略**：
    - 考虑更短的代理任务
    - 或更小的模型仅用于调优
    - 缩小范围

    #### 4. 记录一切 📝

    **如前所述**：
    - 记录每次试验的超参数和结果
    - 用于可重现性
    - 如果需要，可以重新访问

    #### 5. 警惕随机性 🎲

    **高方差训练**：
    - 相同超参数的不同运行产生不同结果
    - 由于随机初始化等

    **解决方案**：
    - 可能需要在HPO期间增加训练确定性
    - 否则搜索可能有噪声
    - 固定随机种子
    - 多次运行并平均
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 💻 HPO示例：使用Scikit-learn

    让我们通过一个简单的示例来说明超参数调优的概念。

    ### 场景：乳腺癌分类

    我们将使用随机森林分类器，并使用RandomizedSearchCV进行超参数调优。
    """
    )
    return


@app.cell
def _():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_predict
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
    from scipy import stats as st
    import numpy as np

    print("💻 超参数调优示例：乳腺癌分类")
    print("=" * 60)

    # 1. 加载数据
    cancer_data = load_breast_cancer()
    X_cancer = cancer_data.data
    y_cancer = cancer_data.target

    print(f"\n📊 数据集信息:")
    print(f"   样本数: {X_cancer.shape[0]}")
    print(f"   特征数: {X_cancer.shape[1]}")
    print(f"   类别: {cancer_data.target_names}")
    print(f"   类别分布: {np.bincount(y_cancer)}")

    # 2. 分割数据
    X_train_hpo, X_test_hpo, y_train_hpo, y_test_hpo = train_test_split(
        X_cancer, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer
    )

    print(f"\n✂️ 数据分割:")
    print(f"   训练集: {X_train_hpo.shape}")
    print(f"   测试集: {X_test_hpo.shape}")

    # 3. 定义基础模型
    rf_base = RandomForestClassifier(n_jobs=-1, random_state=42)

    # 4. 定义超参数搜索空间
    param_distributions_demo = {
        'n_estimators': st.randint(100, 600),
        'max_depth': st.randint(3, 21),
        'max_features': st.uniform(0.3, 0.7),
        'min_samples_split': st.randint(2, 11),
        'min_samples_leaf': st.randint(1, 5),
        'bootstrap': [True, False],
        'class_weight': [None, 'balanced']
    }

    print(f"\n🔧 超参数搜索空间:")
    for p_name, p_dist in param_distributions_demo.items():
        print(f"   {p_name}: {p_dist}")

    # 5. 定义交叉验证策略
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n🔍 交叉验证策略: 5折分层K折")

    return RandomForestClassifier, RandomizedSearchCV, StratifiedKFold, X_cancer, X_test_hpo, X_train_hpo, accuracy_score, cancer_data, classification_report, cross_val_predict, cv_strategy, load_breast_cancer, np, p_dist, p_name, param_distributions_demo, rf_base, roc_auc_score, st, train_test_split, y_cancer, y_test_hpo, y_train_hpo


@app.cell
def _(RandomizedSearchCV, X_test_hpo, X_train_hpo, accuracy_score, classification_report, cv_strategy, np, param_distributions_demo, rf_base, roc_auc_score, y_test_hpo, y_train_hpo):
    print("\n🚀 开始随机搜索...")
    print("   (这可能需要几分钟)")

    # 创建随机搜索对象
    random_search_demo = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_distributions_demo,
        n_iter=20,  # 减少迭代次数以加快演示
        scoring='roc_auc',
        cv=cv_strategy,
        random_state=42,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        verbose=0
    )

    # 执行搜索
    random_search_demo.fit(X_train_hpo, y_train_hpo)

    print(f"\n✅ 搜索完成！")
    print(f"\n🏆 最佳超参数:")
    for best_param_name, best_param_value in random_search_demo.best_params_.items():
        print(f"   {best_param_name}: {best_param_value}")

    print(f"\n📈 最佳交叉验证得分: {random_search_demo.best_score_:.4f}")

    # 获取最佳模型
    best_model_demo = random_search_demo.best_estimator_

    # 在测试集上评估
    y_pred_proba_demo = best_model_demo.predict_proba(X_test_hpo)[:, 1]
    y_pred_demo = best_model_demo.predict(X_test_hpo)

    test_auc_demo = roc_auc_score(y_test_hpo, y_pred_proba_demo)
    test_acc_demo = accuracy_score(y_test_hpo, y_pred_demo)

    print(f"\n🎯 测试集性能:")
    print(f"   ROC AUC: {test_auc_demo:.4f}")
    print(f"   准确率: {test_acc_demo:.4f}")

    print(f"\n📊 分类报告:")
    print(classification_report(y_test_hpo, y_pred_demo, target_names=['恶性', '良性']))

    # 显示前5个最佳配置
    results_df_demo = random_search_demo.cv_results_
    top_5_indices = np.argsort(results_df_demo['rank_test_score'])[:5]

    print(f"\n🔝 前5个最佳配置:")
    for rank_idx, config_idx in enumerate(top_5_indices, 1):
        mean_score = results_df_demo['mean_test_score'][config_idx]
        std_score = results_df_demo['std_test_score'][config_idx]
        print(f"   {rank_idx}. 得分: {mean_score:.4f} (+/- {std_score:.4f})")

    return best_model_demo, best_param_name, best_param_value, config_idx, mean_score, random_search_demo, rank_idx, results_df_demo, std_score, test_acc_demo, test_auc_demo, top_5_indices, y_pred_demo, y_pred_proba_demo


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 关键观察 🔍

    #### 1. 随机搜索效率

    - 只尝试了20个配置（而不是所有可能的组合）
    - 通常能找到接近最优的解决方案
    - 比网格搜索快得多

    #### 2. 交叉验证的重要性

    - 使用5折交叉验证评估每个配置
    - 减少过拟合到特定验证集的风险
    - 提供更可靠的性能估计

    #### 3. 测试集保持不变

    - 测试集从未用于超参数选择
    - 提供无偏的泛化性能估计
    - 这是关键的最佳实践

    #### 4. 避免过拟合到验证集

    **问题**：如果运行大量搜索，可能会偶然过拟合到验证集

    **理解**：
    - 每个超参数组合实际上是一个新模型
    - 尝试数百或数千个组合
    - 某些组合可能仅凭运气在验证集上看起来很好
    - 即使它们不会泛化

    **解决策略**：
    - 使用交叉验证（我们已经在做）
    - 考虑嵌套交叉验证（计算密集）
    - 保持最终测试集不变
    - 意识到验证性能可能略微乐观
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🤖 软AutoML vs 硬AutoML

    ### 软AutoML

    **我们刚才描述的过程**：
    - 调整超参数
    - 自动化模型构建的一个方面

    **特点**：
    - 相对简单
    - 计算成本可控
    - 人类仍然选择模型类型和特征

    ### 硬AutoML

    **更进一步**：
    - 尝试自动化模型架构设计
    - 自动化特征工程

    **包括**：
    - 神经架构搜索（NAS）
    - 自动尝试不同的网络架构
    - 特征选择算法
    - 自动选择最佳特征子集

    **示例**：
    - Google的AutoML
    - 通过在架构空间中进行广泛搜索
    - 产生了像EfficientNet这样的架构

    ### 权衡

    **硬AutoML**：
    - 计算成本高
    - 功能强大
    - 可能无法捕获所有领域知识

    **实践中**：
    - 完全AutoML（按按钮，获得完整模型管道）是一个吸引人的想法
    - 但通常需要大量计算
    - 可能无法捕获人类可以注入的所有领域知识

    **最佳方法**：
    - 人类直觉引导的手动探索
    - 结合自动超参数调优
    - 通常提供最佳权衡
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🎓 本章总结

    我们已经涵盖了本章的核心概念。在下一章中，我们将继续讨论模型优化，然后是压缩技术和其他高级主题。

    ### 关键要点

    #### 1. 从简单开始，智能迭代 🚀
    - 不要直接跳到最花哨的模型
    - 使用启发式和简单模型验证管道和假设
    - 逐步增强复杂度（阶段1 → 4）
    - 这种分阶段方法确保你在坚实的基础上构建
    - 通常产生更可维护的系统

    #### 2. 严格实验并跟踪一切 📝
    - 记录所有实验
    - 跟踪超参数和结果
    - 使用版本控制
    - 确保可重现性

    #### 3. 通过超参数调优优化 🎛️
    - 使用自动搜索方法
    - 随机搜索通常是一个好的起点
    - 考虑贝叶斯优化用于昂贵的训练
    - 始终使用交叉验证

    #### 4. 保持测试集不变 🔒
    - 获得诚实的性能估计
    - 永远不要在测试集上调优
    - 测试集仅用于最终评估

    #### 5. 调试是一项技能 🐛
    - 结合软件调试和科学调试
    - 从简单开始，逐步增加复杂度
    - 监控训练指标
    - 检查数据质量

    ### 下一步

    在下一部分中，我们将继续探讨与MLOps周期建模阶段相关的更多高级概念：

    - 模型压缩技术
    - 迁移学习和微调
    - 模型部署策略
    - CI/CD工作流
    - 监控和观测
    - LLMOps特殊考虑

    ### 核心理念

    **系统思维**：将机器学习视为更广泛软件生态系统的一个活的部分，而不是独立的工件。

    目标始终是帮助你培养成熟的、以系统为中心的思维方式。🚀
    """
    )
    return


@app.cell
def _():
    from datetime import datetime
    return (datetime,)


if __name__ == "__main__":
    app.run()

