import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 简介""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    目标：使用 OpenR1 的数学数据集，通过 GRPO 将 `Qwen3-4B-Base` 转换为推理模型。

    我们首先对模型进行预微调，使 GRPO 跳过尝试匹配格式 - 这会加速 GRPO 训练。
    """
    )
    return


@app.cell
def _():
    from unsloth import FastLanguageModel
    import torch
    max_seq_length = 2048 # 可以增加以支持更长的推理轨迹
    lora_rank = 32 # 更大的 rank = 更智能，但更慢

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-4B-Base",
        max_seq_length = max_seq_length,
        load_in_4bit = False, # LoRA 16bit 设置为 False
        fast_inference = False, # 启用 vLLM 快速推理
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.7, # 如果内存不足请减小
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # 选择任何 > 0 的数字！建议 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = lora_rank*2, # *2 加速训练
        use_gradient_checkpointing = "unsloth", # 减少内存使用
        random_state = 3407,
    )
    return max_seq_length, model, tokenizer, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### GRPO 对话模板
    由于我们使用的是基础模型，应该设置一个对话模板。你也可以创建自己的对话模板！
    1. DeepSeek 使用 `<think>` 和 `</think>`，但这**不是**必需的 - 你可以随意自定义！
    2. 建议使用 `system_prompt` 至少引导模型的响应。
    """
    )
    return


@app.cell
def _():
    reasoning_start = "<think>"
    reasoning_end   = "</think>"
    solution_start  = "<SOLUTION>"
    solution_end    = "</SOLUTION>"

    system_prompt = \
    f"""你会得到一个问题。
    思考这个问题并提供你的解题过程。
    将其放在 {reasoning_start} 和 {reasoning_end} 之间。
    然后，在 {solution_start}{solution_end} 之间提供你的解决方案"""
    system_prompt
    return (
        reasoning_end,
        reasoning_start,
        solution_end,
        solution_start,
        system_prompt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""我们在下面创建一个简单的对话模板。注意 `add_generation_prompt` 包括在前面添加 `<think>` 以引导模型开始其推理过程。""")
    return


@app.cell
def _(reasoning_start, system_prompt, tokenizer):
    chat_template = \
        "{% if messages[0]['role'] == 'system' %}"\
            "{{ messages[0]['content'] + eos_token }}"\
            "{% set loop_messages = messages[1:] %}"\
        "{% else %}"\
            "{{ '{system_prompt}' + eos_token }}"\
            "{% set loop_messages = messages %}"\
        "{% endif %}"\
        "{% for message in loop_messages %}"\
            "{% if message['role'] == 'user' %}"\
                "{{ message['content'] }}"\
            "{% elif message['role'] == 'assistant' %}"\
                "{{ message['content'] + eos_token }}"\
            "{% endif %}"\
        "{% endfor %}"\
        "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
        "{% endif %}"

    # Replace with out specific template:
    chat_template = chat_template\
        .replace("'{system_prompt}'",   f"'{system_prompt}'")\
        .replace("'{reasoning_start}'", f"'{reasoning_start}'")
    tokenizer.chat_template = chat_template
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""让我们看看对话模板在示例上的表现：""")
    return


@app.cell
def _(reasoning_end, reasoning_start, solution_end, solution_start, tokenizer):
    tokenizer.apply_chat_template([
        {"role" : "user", "content" : "1+1 等于多少？"},
        {"role" : "assistant", "content" : f"{reasoning_start}我认为是 2。{reasoning_end}{solution_start}2{solution_end}"},
        {"role" : "user", "content" : "2+2 等于多少？"},
    ], tokenize = False, add_generation_prompt = True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 格式化预微调
    我们现在使用 NVIDIA 的 [Open Math Reasoning 数据集](https://huggingface.co/datasets/nvidia/OpenMathReasoning) 的一个子集，该子集已过滤为仅包含高质量的 DeepSeek R1 轨迹。

    我们只会过滤约 59 个示例，首先"启动"/预微调模型以理解我们的自定义 GRPO 格式。
    """
    )
    return


@app.cell
def _():
    from datasets import load_dataset
    import pandas as pd
    import numpy as np

    dataset = load_dataset("unsloth/OpenMathReasoning-mini", split = "cot")
    dataset = dataset.to_pandas()[
        ["expected_answer", "problem", "generated_solution"]
    ]

    # 尝试转换为数字 - 如果不行，替换为 NaN
    is_number = pd.to_numeric(pd.Series(dataset["expected_answer"]), errors = "coerce").notnull()
    # 只选择数字
    dataset = dataset.iloc[np.where(is_number)[0]]

    dataset
    return dataset, load_dataset, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""我们必须格式化数据集以遵循我们的 GRPO 风格格式：""")
    return


@app.cell
def _(
    dataset,
    reasoning_end,
    reasoning_start,
    solution_end,
    solution_start,
    system_prompt,
):
    def format_dataset(x):
        expected_answer = x["expected_answer"]
        problem = x["problem"]

        # 移除生成的 <think> 和 </think>
        thoughts = x["generated_solution"]
        thoughts = thoughts.replace("<think>", "").replace("</think>", "")

        # 去除左右两侧的换行符
        thoughts = thoughts.strip()
        # 添加我们的自定义格式
        final_prompt = \
            reasoning_start + thoughts + reasoning_end + \
            solution_start + expected_answer + solution_end
        return [
            {"role" : "system",    "content" : system_prompt},
            {"role" : "user",      "content" : problem},
            {"role" : "assistant", "content" : final_prompt},
        ]

    dataset["Messages"] = dataset.apply(format_dataset, axis = 1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""检查是否成功：""")
    return


@app.cell
def _(dataset, tokenizer):
    tokenizer.apply_chat_template(dataset["Messages"][0], tokenize = False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    让我们将预微调数据集截断为 `max_seq_length/2`，因为我们不想要太长的推理轨迹。

    注意这可能需要 2 分钟！
    """
    )
    return


@app.cell
def _(dataset, max_seq_length, tokenizer):
    dataset['N'] = dataset['Messages'].apply(lambda x: len(tokenizer.apply_chat_template(x)))
    dataset_1 = dataset.loc[dataset['N'] <= max_seq_length / 2].copy()
    dataset_1.shape
    return (dataset_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""然后我们对消息进行分词并将其转换为 Hugging Face 兼容的数据集格式：""")
    return


@app.cell
def _(dataset_1, tokenizer):
    from datasets import Dataset
    dataset_1['text'] = tokenizer.apply_chat_template(dataset_1['Messages'].values.tolist(), tokenize=False)
    dataset_2 = Dataset.from_pandas(dataset_1)
    dataset_2
    return (dataset_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""现在让我们预微调模型，使其遵循我们的自定义 GRPO 格式！""")
    return


@app.cell
def _(dataset_2, model, tokenizer):
    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset_2, args=SFTConfig(dataset_text_field='text', per_device_train_batch_size=1, gradient_accumulation_steps=1, warmup_steps=5, num_train_epochs=2, learning_rate=0.0002, logging_steps=5, optim='adamw_8bit', weight_decay=0.01, lr_scheduler_type='linear', seed=3407, report_to='none'))  # 使用 GA 模拟批次大小！  # 设置为 1 进行完整训练  # 长时间训练时减少到 2e-5  # 用于 WandB 等
    return (trainer,)


@app.cell
def _(trainer):
    trainer.train()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""让我们检查模型是否学会了遵循自定义格式：""")
    return


@app.cell
def _(dataset_2, model, tokenizer):
    _text = tokenizer.apply_chat_template(dataset_2[0]['Messages'][:2], tokenize=False, add_generation_prompt=True)
    from transformers import TextStreamer
    _ = model.generate(**tokenizer(_text, return_tensors='pt').to('cuda'), temperature=0, max_new_tokens=1024, streamer=TextStreamer(tokenizer, skip_prompt=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""是的，它确实遵循了格式！太好了！让我们在 GRPO 步骤之前删除一些项目""")
    return


@app.cell
def _(dataset_2, torch):
    del dataset_2
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 数据准备
    <a name="Data"></a>

    我们使用 Hugging Face 的 [Open R1 Math 数据集](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed)。你也可以使用 OpenAI 著名的 [GSM8K 数据集](https://huggingface.co/datasets/openai/gsm8k)
    """
    )
    return


@app.cell
def _(load_dataset):
    dataset_3 = load_dataset('open-r1/DAPO-Math-17k-Processed', 'en', split='train')
    dataset_3
    return (dataset_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""让我们看看第一行：""")
    return


@app.cell
def _(dataset_3):
    dataset_3[0]['prompt']
    return


@app.cell
def _(dataset_3):
    dataset_3[0]['solution']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""在 GSM8K 中，我们注意到所有答案都有 ####，所以我们提取它。但对于 Open R1 数据集，我们可以跳过下面的步骤。""")
    return


@app.cell
def _(dataset_3):
    def extract_hash_answer(text):
        return text
    extract_hash_answer(dataset_3[0]['solution'])
    return (extract_hash_answer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""让我们映射数据集！并查看第一行：""")
    return


@app.cell
def _(dataset_3, extract_hash_answer, system_prompt):
    dataset_4 = dataset_3.map(lambda x: {'prompt': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': x['prompt']}], 'answer': extract_hash_answer(x['solution'])})
    dataset_4[0]
    return (dataset_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""我们创建一个正则表达式格式来匹配推理部分和答案：""")
    return


@app.cell
def _(reasoning_end, solution_start, tokenizer):
    import re

    # 添加可选的 EOS token 匹配
    solution_end_regex = r"</SOLUTION>[\s]{0,}" + \
        "(?:" + re.escape(tokenizer.eos_token) + ")?"

    match_format = re.compile(
        rf"{reasoning_end}.*?"\
        rf"{solution_start}(.+?){solution_end_regex}"\
        rf"[\s]{{0,}}$",
        flags = re.MULTILINE | re.DOTALL
    )
    match_format
    return match_format, re


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""我们验证它是否有效：""")
    return


@app.cell
def _(match_format):
    match_format.findall(
        "Let me think!</think>"\
        f"<SOLUTION>\n2\n</SOLUTION>",
    )
    return


@app.cell
def _(match_format):
    match_format.findall(
        "<think>Let me think!</think>"\
        f"<SOLUTION>  2  </SOLUTION>\n\n",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""现在我们想创建一个奖励函数来精确匹配格式 - 如果成功，我们奖励 3 分：""")
    return


@app.cell
def _(match_format):
    def match_format_exactly(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]['content']
            if match_format.search(response) is not None:  # 如果格式完全匹配则奖励！
                score = score + 3.0
            scores.append(score)
        return scores
    return (match_format_exactly,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""如果失败，我们希望在模型至少部分遵循格式时奖励它，通过计算每个符号：""")
    return


@app.cell
def _(reasoning_end, solution_end, solution_start):
    def match_format_approximately(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]['content']
            score = score + (0.5 if response.count(reasoning_end) == 1 else -1.0)  # 计算看到多少关键字 - 如果太多则惩罚！
            score = score + (0.5 if response.count(solution_start) == 1 else -1.0)  # 如果看到 1 个，则加一些分！
            score = score + (0.5 if response.count(solution_end) == 1 else -1.0)
            scores.append(score)  # 不需要奖励 <start_working_out>，因为我们总是在前面添加它！
        return scores  # score += 0.5 if response.count(reasoning_start) == 1 else -1.0
    return (match_format_approximately,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""最后，我们想提取生成的答案，并奖励或惩罚它！我们还根据答案与真实答案的接近程度通过比率来奖励：""")
    return


@app.cell
def _(match_format):
    def check_answer(prompts, completions, answer, **kwargs):
        question = prompts[0][-1]['content']
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [guess.group(1) if (guess := match_format.search(r)) is not None else None for r in responses]
        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            score = 0
            if guess is None:
                scores.append(-2.0)
                continue
            if guess == true_answer:
                score = score + 5.0
            elif guess.strip() == true_answer.strip():
                score = score + 3.5
            else:
                try:
                    ratio = float(guess) / float(true_answer)  # 正确答案得 5 分！
                    if ratio >= 0.9 and ratio <= 1.1:
                        score = score + 2.0
                    elif ratio >= 0.8 and ratio <= 1.2:  # 如果看到空格则匹配，但奖励较少
                        score = score + 1.5
                    else:
                        score = score - 2.5
                except:  # 如果答案通过比率接近，我们也会奖励！
                    score = score - 4.5  # 即如果答案在某个范围内，则奖励！
            scores.append(score)
        return scores  # 惩罚错误答案  # 惩罚
    return (check_answer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    有时答案可能不是 1 个数字，而是像一个句子，例如"解决方案是 $20" -> 我们提取 20。

    我们还删除可能的逗号，例如 123,456
    """
    )
    return


@app.cell
def _(re, solution_start):
    match_numbers = re.compile(
        solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
        flags = re.MULTILINE | re.DOTALL
    )
    print(match_numbers.findall("<SOLUTION>  0.34  </SOLUTION>"))
    print(match_numbers.findall("<SOLUTION>  123,456  </SOLUTION>"))
    print(match_numbers.findall("<SOLUTION>  -0.234  </SOLUTION>"))
    print(match_numbers.findall("<SOLUTION>17</SOLUTION>"))
    return (match_numbers,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""我们现在准备主函数，它将打印生成的响应和真实答案，以及另一个奖励函数，该函数通过 `float` 将文本转换为浮点数并查看是否相同。""")
    return


@app.cell
def _(match_numbers):
    global PRINTED_TIMES
    PRINTED_TIMES = 0
    global PRINT_EVERY_STEPS
    PRINT_EVERY_STEPS = 5

    def check_numbers(prompts, completions, answer, **kwargs):
        question = prompts[0][-1]['content']
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [guess.group(1) if (guess := match_numbers.search(r)) is not None else None for r in responses]
        scores = []
        global PRINTED_TIMES
        global PRINT_EVERY_STEPS
        if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
            print('*' * 20 + f'问题:\n{question}', f'\n答案:\n{answer[0]}', f'\n响应:\n{responses[0]}', f'\n提取:\n{extracted_responses[0]}')
        PRINTED_TIMES = PRINTED_TIMES + 1
        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:  # 只每隔几步打印一次
                scores.append(-2.5)
                continue
            try:
                true_answer = float(true_answer.strip())
                guess = float(guess.strip().replace(',', ''))
                scores.append(3.5 if guess == true_answer else -1.5)
            except:
                scores.append(0)
                continue
        return scores  # 转换为数字  # 删除逗号，如 123,456
    return PRINTED_TIMES, PRINT_EVERY_STEPS, check_numbers


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    获取前 90% 的提示长度，这样我们就不会意外截断它们！

    即我们将删除前 10% 的长提示。
    """
    )
    return


@app.cell
def _(dataset_4, np, tokenizer):
    tokenized = dataset_4.map(lambda x: {'tokens': tokenizer.apply_chat_template(x['prompt'], add_generation_prompt=True, tokenize=True)}, batched=True)
    print(tokenizer.decode(tokenized[0]['tokens']))
    tokenized = tokenized.map(lambda x: {'L': len(x['tokens'])})
    maximum_length = int(np.quantile(tokenized['L'], 0.9))
    print('最大长度 = ', maximum_length)
    dataset_5 = dataset_4.select(np.where(np.array(tokenized['L']) <= maximum_length)[0])
    # 只过滤小于 90% 最大长度的样本
    del tokenized
    return dataset_5, maximum_length


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <a name="Train"></a>
    ### 训练模型

    现在设置 GRPO Trainer 和所有配置！
    """
    )
    return


@app.cell
def _(max_seq_length, maximum_length, tokenizer):
    max_prompt_length = maximum_length + 1 # + 1 以防万一！
    max_completion_length = max_seq_length - max_prompt_length

    from vllm import SamplingParams
    vllm_sampling_params = SamplingParams(
        min_p = 0.1,
        top_p = 1.0,
        top_k = -1,
        seed = 3407,
        stop = [tokenizer.eos_token],
        include_stop_str_in_output = True,
    )

    from trl import GRPOConfig, GRPOTrainer
    training_args = GRPOConfig(
        vllm_sampling_params = vllm_sampling_params,
        temperature = 1.0,
        learning_rate = 5e-6,
        weight_decay = 0.01,
        warmup_ratio = 0.1,
        lr_scheduler_type = "linear",
        optim = "adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1, # 增加到 4 以获得更平滑的训练
        num_generations = 4, # 如果内存不足请减少
        max_prompt_length = max_prompt_length,
        max_completion_length = max_completion_length,
        # num_train_epochs = 1, # 设置为 1 进行完整训练
        max_steps = 100,
        save_steps = 100,
        report_to = "none", # 可以使用 Weights & Biases
        output_dir = "outputs",

        # 可选的训练 + 评估
        # fp16_full_eval = True,
        # per_device_eval_batch_size = 4,
        # eval_accumulation_steps = 1,
        # eval_strategy = "steps",
        # eval_steps = 1,
    )
    return GRPOTrainer, SamplingParams, training_args


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    让我们运行训练器！如果你向上滚动，你会看到一个奖励表。目标是看到 `reward` 列增加！

    你可能需要等待 150 到 200 步才能看到任何效果。前 100 步你可能会得到 0 奖励。请耐心等待！

    | Step | Training Loss | reward    | reward_std | completion_length | kl       |
    |------|---------------|-----------|------------|-------------------|----------|
    | 1    | 0.000000      | 0.125000  | 0.000000   | 200.000000        | 0.000000 |
    | 2    | 0.000000      | 0.072375  | 0.248112   | 200.000000        | 0.000000 |
    | 3    | 0.000000      | -0.079000 | 0.163776   | 182.500000        | 0.000005 |
    """
    )
    return


@app.cell
def _(
    GRPOTrainer,
    check_answer,
    check_numbers,
    dataset_5,
    match_format_approximately,
    match_format_exactly,
    model,
    tokenizer,
    training_args,
):
    # 可选的训练 + 评估
    # new_dataset = dataset.train_test_split(test_size = 0.01)
    trainer_1 = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=[match_format_exactly, match_format_approximately, check_answer, check_numbers], args=training_args, train_dataset=dataset_5)
    trainer_1.train()  # 可选的训练 + 评估  # train_dataset = new_dataset["train"],  # eval_dataset = new_dataset["test"],
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <a name="Inference"></a>
    ### 推理
    现在让我们试试刚刚训练的模型！首先，让我们先试试没有经过 GRPO 训练的模型：
    """
    )
    return


@app.cell
def _(SamplingParams, model):
    _text = '101 的平方根是多少？'
    _sampling_params = SamplingParams(temperature=1.0, top_k=50, max_tokens=1024)
    _output = model.fast_generate([_text], sampling_params=_sampling_params, lora_request=None)[0].outputs[0].text
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""现在使用我们刚刚用 GRPO 训练的 LoRA - 我们首先保存 LoRA！""")
    return


@app.cell
def _(model):
    model.save_lora("grpo_saved_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""验证 LoRA 确实已训练！""")
    return


@app.cell
def _():
    from safetensors import safe_open

    tensors = {}
    with safe_open("grpo_saved_lora/adapter_model.safetensors", framework = "pt") as f:
        # 验证 A 和 B 都不为零
        for key in f.keys():
            tensor = f.get_tensor(key)
            n_zeros = (tensor == 0).sum() / tensor.numel()
            assert(n_zeros.item() != tensor.numel())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""现在我们加载 LoRA 并测试：""")
    return


@app.cell
def _(SamplingParams, model, system_prompt, tokenizer):
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': '101 的平方根是多少？'}]
    _text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    _sampling_params = SamplingParams(temperature=1.0, top_k=50, max_tokens=2048)
    _output = model.fast_generate(_text, sampling_params=_sampling_params, lora_request=model.load_lora('grpo_saved_lora'))[0].outputs[0].text
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""我们的推理模型好多了 - 它并不总是正确的，因为我们只训练了大约一个小时 - 如果我们延长序列长度并训练更长时间，它会更好！""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
