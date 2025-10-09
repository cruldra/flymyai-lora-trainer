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
    # GRPO (Group Relative Policy Optimization) 完整笔记

    本笔记整理了关于 GRPO 的全面资料，包括理论基础、与其他算法的对比、实际应用等。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1. GRPO 简介

    **Group Relative Policy Optimization (GRPO)** 是一种专为增强大语言模型推理能力而设计的强化学习算法。

    ### 核心特点

    - **无需价值函数**: 与 PPO 不同，GRPO 不需要单独的价值网络（Critic），大幅降低内存和计算开销（约减少 50%）
    - **组相对优势**: 通过为每个提示生成多个响应，使用组内平均奖励作为基线
    - **确定性奖励**: 可以使用简单的规则（如正则表达式）而非神经网络奖励模型
    - **内存效率**: 在 16GB VRAM 上即可训练 1B 参数的推理模型

    ### 首次应用

    - **DeepSeekMath** (2024年2月): 首次引入 GRPO 用于数学推理
    - **DeepSeek-R1** (2025年1月): 使用 GRPO 实现与 OpenAI o1 竞争的推理能力
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. GRPO 工作原理

    ### 基本流程

    1. **生成多个响应**: 对每个提示 $p$，生成 $N$ 个响应 $\mathcal{G}=\{r_1, r_2,...r_N\}$
    2. **计算奖励**: 使用奖励模型 $R_\phi$ 为每个响应分配奖励
    3. **计算组归一化优势**:
       $$A_i = \frac{R_\phi(r_i) - \text{mean}(\mathcal{G})}{\text{std}(\mathcal{G})}$$
    4. **更新策略**: 使用优势值更新模型参数

    ### 优势计算详解

    优势（Advantage）定义了特定动作相比平均动作的好坏程度：

    - 如果 $A_i > 0$: 该响应优于平均水平，应该被强化
    - 如果 $A_i < 0$: 该响应劣于平均水平，应该被抑制
    - 如果 $A_i \approx 0$: 该响应接近平均水平

    通过标准化（减去均值除以标准差），我们得到一个以 0 为中心的分布，便于模型学习。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. GRPO vs PPO 对比

    ### PPO (Proximal Policy Optimization)

    PPO 是 ChatGPT 使用的强化学习算法，需要以下组件：

    1. **Policy Model** ($\pi_\theta$): 正在训练的 LLM
    2. **Reference Model** ($\pi_{\theta_{old}}$): 冻结的原始模型副本
    3. **Reward Model** ($R_\phi$): 预测人类偏好的模型
    4. **Value Model/Critic** ($V_\gamma$): 估计长期奖励的模型

    ### GRPO 的简化

    GRPO 只需要：

    1. **Policy Model** ($\pi_\theta$): 正在训练的 LLM
    2. **Reference Model** ($\pi_{\theta_{old}}$): 冻结的原始模型副本
    3. **Reward Function**: 可以是简单的规则或神经网络

    **关键区别**: GRPO 移除了 Value Model，通过组内采样来估计优势。

    ### 内存和计算对比

    | 算法 | 需要的模型数量 | 可训练参数 | 相对内存使用 |
    |------|---------------|-----------|-------------|
    | PPO  | 4个LLM        | 2个模型   | 100%        |
    | GRPO | 2个LLM        | 1个模型   | ~50%        |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4. GRPO 数学公式

    ### 损失函数

    GRPO 的目标函数包含两个主要部分：

    $$\mathcal{L}_{\text{GRPO}}(\theta) = \mathcal{L}_{\text{clip}}(\theta) - w_1\mathbb{D}_{\text{KL}}(\pi_\theta || \pi_{\text{orig}})$$

    #### 4.1 裁剪代理损失 (Clipped Surrogate Loss)

    $$\mathcal{L}_{\text{clip}}(\theta) = \frac{1}{N} \sum_{i=1}^N \min\left( \frac{\pi_\theta(r_i|p)}{\pi_{\theta_{\text{old}}}(r_i|p)} A_i, \text{clip}\left( \frac{\pi_\theta(r_i|p)}{\pi_{\theta_{\text{old}}}(r_i|p)}, 1-\epsilon, 1+\epsilon \right) A_i \right)$$

    其中：
    - $\epsilon$ 控制裁剪范围（通常为 0.2）
    - 裁剪防止策略更新过大，保持训练稳定性

    #### 4.2 KL 散度惩罚

    $$\text{KL}(\theta) = \mathbb{E}_{s_t} \left[ \mathbb{D}_{\text{KL}}(\pi_{\theta\text{orig}}(\cdot | s_t) || \pi_{\theta}(\cdot | s_t)) \right]$$

    作用：防止新策略偏离原始模型太远，保持生成文本的连贯性。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. 奖励函数设计

    ### DeepSeek-R1 的确定性奖励

    DeepSeek-R1 使用基于规则的奖励函数，而非神经网络：

    #### 5.1 格式奖励

    - **精确格式匹配**: 检查是否包含 `<reasoning>...</reasoning><answer>...</answer>`
    - **近似格式匹配**: 检查是否至少包含 `<reasoning>` 或 `<answer>` 标签

    #### 5.2 答案正确性

    - **完全匹配**: 答案与标准答案完全一致
    - **数字匹配**: 提取响应中的数字，计算与标准答案的重叠度

    #### 5.3 语言一致性

    - 惩罚混合语言输出（如中文问题用英文回答）

    ### 示例代码
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import re

    def exact_format_match(response):
        """精确格式匹配"""
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        return 1.0 if re.search(pattern, response, re.DOTALL) else 0.0

    def check_answer(response, ground_truth):
        """检查答案正确性"""
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if answer_match:
            predicted = answer_match.group(1).strip()
            return 2.0 if predicted == ground_truth else 0.0
        return 0.0

    def compute_reward(response, ground_truth):
        """计算总奖励"""
        reward = 0.0
        reward += exact_format_match(response)
        reward += check_answer(response, ground_truth)
        return reward
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 6. 优势计算示例

    让我们通过一个具体例子来理解优势计算：
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import numpy as np

    # 假设我们有4个响应的奖励
    rewards = np.array([4.0, 2.5, 0.5, 0.1])

    # 计算优势
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    advantages = (rewards - mean_reward) / std_reward

    print(f"奖励: {rewards}")
    print(f"平均奖励: {mean_reward:.2f}")
    print(f"标准差: {std_reward:.2f}")
    print(f"优势: {advantages}")
    print(f"\n解释:")
    for i, (r, a) in enumerate(zip(rewards, advantages)):
        if a > 0:
            print(f"  响应{i+1}: 奖励={r:.1f}, 优势={a:.2f} → 优于平均，应强化")
        else:
            print(f"  响应{i+1}: 奖励={r:.1f}, 优势={a:.2f} → 劣于平均，应抑制")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 7. DeepSeek-R1 训练流程

    DeepSeek-R1 的训练采用多阶段流程，交替使用 SFT 和 GRPO：

    ### 阶段 1: 冷启动 SFT
    - 使用少量高质量数据（几千个样本）
    - 人工验证的推理链
    - 目标：让模型学会基本的推理格式

    ### 阶段 2: GRPO 推理训练
    - 训练模型生成 `<reasoning>...</reasoning>` 推理轨迹
    - 使用确定性奖励：格式、一致性、正确性
    - 目标：提升推理能力

    ### 阶段 3: 合成数据 SFT
    - 生成 80万个合成数据点
    - 使用拒绝采样过滤错误响应
    - LLM-as-a-Judge 评估质量
    - 目标：扩大高质量数据规模

    ### 阶段 4: GRPO 对齐
    - 对齐模型使其有用且无害
    - 目标：最终优化

    ### 性能提升

    在 DeepSeekMath-Instruct 7B 上的结果：

    | 数据集 | 训练前 | 训练后 | 提升 |
    |--------|--------|--------|------|
    | GSM8K  | 82.9%  | 88.2%  | +5.3% |
    | MATH   | 46.8%  | 51.7%  | +4.9% |
    | CMATH  | 84.6%  | 88.8%  | +4.2% |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 8. GRPO vs 其他算法对比表

    | 算法 | 价值函数 | 优势估计方法 | 稳定性机制 | 内存使用 | 适用场景 |
    |------|---------|-------------|-----------|---------|---------|
    | **GRPO** | ❌ 无 | 组均值作为基线 | KL散度惩罚 | 低 | LLM推理任务 |
    | **PPO** | ✅ 有 | 价值函数估计 | 裁剪代理目标 | 高 | 通用RL任务 |
    | **TRPO** | ✅ 有 | 价值函数+信任域 | Hessian信任域 | 高 | 需要稳定性的任务 |
    | **REINFORCE** | ❌ 无 | 无基线 | 无 | 低 | 简单任务 |
    | **DPO** | ❌ 无 | 偏好对比 | 隐式奖励 | 低 | 偏好对齐 |

    ### 关键洞察

    1. **GRPO 的创新**: 用组采样替代价值函数，既保持了低方差，又降低了内存需求
    2. **为什么现在才出现**: 现代 GPU/TPU 使得快速生成多个样本成为可能
    3. **与 REINFORCE 的关系**: GRPO 本质上是改进的 REINFORCE，使用更好的基线（组均值而非单批次均值）
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 9. 实际应用建议

    ### 何时使用 GRPO

    ✅ **适合的场景**:
    - 数学推理任务
    - 代码生成
    - 有明确正确答案的任务
    - GPU 内存受限的环境
    - 需要快速迭代的项目

    ❌ **不适合的场景**:
    - 开放式创意写作
    - 主观性强的任务
    - 难以定义奖励函数的任务

    ### 训练技巧

    1. **组大小选择**: DeepSeek 使用 64 个样本/问题，可根据资源调整
    2. **学习率**: 策略模型使用 1e-6（比 SFT 小）
    3. **KL 系数**: 0.04 是一个好的起点
    4. **批次大小**: 1024（可根据 GPU 内存调整）
    5. **采样策略**: 使用温度采样增加多样性

    ### 硬件需求

    - **1B 模型**: 16GB VRAM（使用 LoRA）
    - **7B 模型**: 40GB VRAM（使用 LoRA）
    - **更大模型**: 多 GPU 或模型并行
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 10. 参考资源

    ### 核心论文

    1. **DeepSeekMath** (2024): [Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300)
    2. **DeepSeek-R1** (2025): [Incentivizing Reasoning Capability in LLMs](https://arxiv.org/abs/2501.xxxxx)

    ### 技术博客

    - [DataCamp: What is GRPO?](https://www.datacamp.com/blog/what-is-grpo-group-relative-policy-optimization)
    - [Oxen.ai: Why GRPO is Important](https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/)
    - [AI Engineering Academy: Theory Behind GRPO](https://aiengineering.academy/LLM/TheoryBehindFinetuning/GRPO/)

    ### 实现库

    - **HuggingFace TRL**: 官方 GRPO 实现
    - **Unsloth**: 高效微调库
    - **DeepSpeed**: 大规模训练优化

    ### 社区讨论

    - Reddit: r/ChatGPTPro - GRPO vs PPO 讨论
    - Twitter: @adithya_s_k - GRPO 实践经验
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 11. 总结

    ### GRPO 的核心优势

    1. **内存效率**: 相比 PPO 减少约 50% 内存使用
    2. **简单性**: 移除价值函数，减少训练复杂度
    3. **有效性**: 在推理任务上达到 SOTA 性能
    4. **可访问性**: 使得个人开发者也能训练推理模型

    ### 未来方向

    - **自适应组大小**: 根据任务难度动态调整采样数量
    - **混合奖励**: 结合规则和神经网络奖励模型
    - **多模态扩展**: 将 GRPO 应用于视觉-语言模型
    - **领域特化**: 针对特定领域（如医疗、法律）优化奖励函数

    ### 最后的思考

    GRPO 体现了 AI 研究的一个重要趋势：**简单性胜过复杂性**。通过移除不必要的组件（价值函数）并利用现代硬件能力（快速采样），GRPO 实现了更高效、更易用的强化学习。

    正如 DeepSeek 团队所展示的，有时候最好的创新不是添加更多功能，而是**停止过度思考，让模型自己学习**。
    """
    )
    return


if __name__ == "__main__":
    app.run()
