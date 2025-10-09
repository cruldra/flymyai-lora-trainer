import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Introduction""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Goal: To convert `Qwen3-4B-Base` into a reasoning model via GRPO by using OpenR1's Math dataset.

    We first pre fine-tune the model to make GRPO skip trying to match formatting - this speeds GRPO up.
    """
    )
    return


@app.cell
def _():
    from unsloth import FastLanguageModel
    import torch
    max_seq_length = 2048 # Can increase for longer reasoning traces
    lora_rank = 32 # Larger rank = smarter, but slower

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-4B-Base",
        max_seq_length = max_seq_length,
        load_in_4bit = False, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.7, # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = lora_rank*2, # *2 speeds up training
        use_gradient_checkpointing = "unsloth", # Reduces memory usage
        random_state = 3407,
    )
    return max_seq_length, model, tokenizer, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### GRPO chat template
    Since we're using a base model, we should set a chat template. You can make your own chat template as well!
    1. DeepSeek uses `<think>` and `</think>`, but this is **not** necessary - you can customize it however you like!
    2. A `system_prompt` is recommended to at least guide the model's responses.
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
    f"""You are given a problem.
    Think about the problem and provide your working out.
    Place it between {reasoning_start} and {reasoning_end}.
    Then, provide your solution between {solution_start}{solution_end}"""
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
    mo.md(r"""We create a simple chat template below. Notice `add_generation_prompt` includes prepending `<think>` to guide the model to start its reasoning process.""")
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
    mo.md(r"""Let's see how our chat template behaves on an example:""")
    return


@app.cell
def _(reasoning_end, reasoning_start, solution_end, solution_start, tokenizer):
    tokenizer.apply_chat_template([
        {"role" : "user", "content" : "What is 1+1?"},
        {"role" : "assistant", "content" : f"{reasoning_start}I think it's 2.{reasoning_end}{solution_start}2{solution_end}"},
        {"role" : "user", "content" : "What is 2+2?"},
    ], tokenize = False, add_generation_prompt = True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Pre fine-tuning for formatting
    We now use a subset of NVIDIA's [Open Math Reasoning dataset](https://huggingface.co/datasets/nvidia/OpenMathReasoning) which was filtered to only include high quality DeepSeek R1 traces.

    We'll only filter ~59 or so examples to first "prime" / pre fine-tune the model to understand our custom GRPO formatting.
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

    # Try converting to number - if not, replace with NaN
    is_number = pd.to_numeric(pd.Series(dataset["expected_answer"]), errors = "coerce").notnull()
    # Select only numbers
    dataset = dataset.iloc[np.where(is_number)[0]]

    dataset
    return dataset, load_dataset, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We have to format the dataset to follow our GRPO style formatting:""")
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

        # Remove generated <think> and </think>
        thoughts = x["generated_solution"]
        thoughts = thoughts.replace("<think>", "").replace("</think>", "")

        # Strip newlines on left and right
        thoughts = thoughts.strip()
        # Add our custom formatting
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
    mo.md(r"""Check to see if it worked:""")
    return


@app.cell
def _(dataset, tokenizer):
    tokenizer.apply_chat_template(dataset["Messages"][0], tokenize = False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let's truncate the pre fine-tuning dataset to `max_seq_length/2` since we don't want too long reasoning traces.

    Note this might take 2 minutes!
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
    mo.md(r"""We then tokenize the messages and convert it to a Hugging Face compatible dataset format:""")
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
    mo.md(r"""Let's now pre fine-tune the model so it follows our custom GRPO formatting!""")
    return


@app.cell
def _(dataset_2, model, tokenizer):
    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset_2, args=SFTConfig(dataset_text_field='text', per_device_train_batch_size=1, gradient_accumulation_steps=1, warmup_steps=5, num_train_epochs=2, learning_rate=0.0002, logging_steps=5, optim='adamw_8bit', weight_decay=0.01, lr_scheduler_type='linear', seed=3407, report_to='none'))  # Use GA to mimic batch size!  # Set this for 1 full training run.  # Reduce to 2e-5 for long training runs  # Use this for WandB etc
    return (trainer,)


@app.cell
def _(trainer):
    trainer.train()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's check if the model has learnt to follow the custom format:""")
    return


@app.cell
def _(dataset_2, model, tokenizer):
    _text = tokenizer.apply_chat_template(dataset_2[0]['Messages'][:2], tokenize=False, add_generation_prompt=True)
    from transformers import TextStreamer
    _ = model.generate(**tokenizer(_text, return_tensors='pt').to('cuda'), temperature=0, max_new_tokens=1024, streamer=TextStreamer(tokenizer, skip_prompt=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Yes it did follow the formatting! Great! Let's remove some items before the GRPO step""")
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
    ### Data Prep
    <a name="Data"></a>

    We're using Hugging Face's [Open R1 Math dataset](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed). You can also utilize OpenAI's famous [GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k)
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
    mo.md(r"""Let's look at the first row:""")
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
    mo.md(r"""In GSM8K, ee notice all answers like about have a ####, so we extract it. But for the Open R1 dataset, we can skip the below.""")
    return


@app.cell
def _(dataset_3):
    def extract_hash_answer(text):
        return _text
    extract_hash_answer(dataset_3[0]['solution'])
    return (extract_hash_answer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's map the dataset! and see the first row:""")
    return


@app.cell
def _(dataset_3, extract_hash_answer, system_prompt):
    dataset_4 = dataset_3.map(lambda x: {'prompt': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': x['prompt']}], 'answer': extract_hash_answer(x['solution'])})
    dataset_4[0]
    return (dataset_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We create a regex format to match the reasoning sections and answers:""")
    return


@app.cell
def _(reasoning_end, solution_start, tokenizer):
    import re

    # Add optional EOS token matching
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
    mo.md(r"""We verify it works:""")
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
    mo.md(r"""We now want to create a reward function to match the format exactly - we reward it with 3 points if it succeeds:""")
    return


@app.cell
def _(match_format):
    def match_format_exactly(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]['content']
            if match_format.search(response) is not None:  # Match if format is seen exactly!
                score = score + 3.0
            scores.append(score)
        return scores
    return (match_format_exactly,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""If it fails, we want to reward the model if it at least follows the format partially, by counting each symbol:""")
    return


@app.cell
def _(reasoning_end, solution_end, solution_start):
    def match_format_approximately(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]['content']
            score = score + (0.5 if response.count(reasoning_end) == 1 else -1.0)  # Count how many keywords are seen - we penalize if too many!
            score = score + (0.5 if response.count(solution_start) == 1 else -1.0)  # If we see 1, then plus some points!
            score = score + (0.5 if response.count(solution_end) == 1 else -1.0)
            scores.append(score)  # No need to reward <start_working_out> since we always prepend it!
        return scores  # score += 0.5 if response.count(reasoning_start) == 1 else -1.0
    return (match_format_approximately,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Finally, we want to extract the generated answer, and reward or penalize it! We also reward it based on how close the answer is to the true one via ratios:""")
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
                    ratio = float(guess) / float(true_answer)  # Correct answer gets 5 points!
                    if ratio >= 0.9 and ratio <= 1.1:
                        score = score + 2.0
                    elif ratio >= 0.8 and ratio <= 1.2:  # Match if spaces are seen, but less reward
                        score = score + 1.5
                    else:
                        score = score - 2.5
                except:  # We also reward it if the answer is close via ratios!
                    score = score - 4.5  # Ie if the answer is within some range, reward it!
            scores.append(score)
        return scores  # Penalize wrong answers  # Penalize
    return (check_answer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Also sometimes it might not be 1 number as the answer, but like a sentence for example "The solution is $20" -> we extract 20.

    We also remove possible commas for example as in 123,456
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
    mo.md(r"""We now prepare our main function which will print out the generated responses and the true answer, along with another reward function which converts text to float via `float` and sees if it's the same.""")
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
            print('*' * 20 + f'Question:\n{question}', f'\nAnswer:\n{answer[0]}', f'\nResponse:\n{responses[0]}', f'\nExtracted:\n{extracted_responses[0]}')
        PRINTED_TIMES = PRINTED_TIMES + 1
        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:  # Print only every few steps
                scores.append(-2.5)
                continue
            try:
                true_answer = float(true_answer.strip())
                guess = float(guess.strip().replace(',', ''))
                scores.append(3.5 if guess == true_answer else -1.5)
            except:
                scores.append(0)
                continue
        return scores  # Convert to numbers  # Remove commas like in 123,456
    return PRINTED_TIMES, PRINT_EVERY_STEPS, check_numbers


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Get the top 90% prompt length so we don't accidentally truncate them!

    Ie we'll remove the top 10% long prompts.
    """
    )
    return


@app.cell
def _(dataset_4, np, tokenizer):
    tokenized = dataset_4.map(lambda x: {'tokens': tokenizer.apply_chat_template(x['prompt'], add_generation_prompt=True, tokenize=True)}, batched=True)
    print(tokenizer.decode(tokenized[0]['tokens']))
    tokenized = tokenized.map(lambda x: {'L': len(x['tokens'])})
    maximum_length = int(np.quantile(tokenized['L'], 0.9))
    print('Max Length = ', maximum_length)
    dataset_5 = dataset_4.select(np.where(np.array(tokenized['L']) <= maximum_length)[0])
    # Filter only samples smaller than 90% max length
    del tokenized
    return dataset_5, maximum_length


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <a name="Train"></a>
    ### Train the model

    Now set up GRPO Trainer and all configurations!
    """
    )
    return


@app.cell
def _(max_seq_length, maximum_length, tokenizer):
    max_prompt_length = maximum_length + 1 # + 1 just in case!
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
        gradient_accumulation_steps = 1, # Increase to 4 for smoother training
        num_generations = 4, # Decrease if out of memory
        max_prompt_length = max_prompt_length,
        max_completion_length = max_completion_length,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps = 100,
        save_steps = 100,
        report_to = "none", # Can use Weights & Biases
        output_dir = "outputs",

        # For optional training + evaluation
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
    And let's run the trainer! If you scroll up, you'll see a table of rewards. The goal is to see the `reward` column increase!

    You might have to wait 150 to 200 steps for any action. You'll probably get 0 reward for the first 100 steps. Please be patient!

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
    # For optional training + evaluation
    # new_dataset = dataset.train_test_split(test_size = 0.01)
    trainer_1 = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=[match_format_exactly, match_format_approximately, check_answer, check_numbers], args=training_args, train_dataset=dataset_5)
    trainer_1.train()  # For optional training + evaluation  # train_dataset = new_dataset["train"],  # eval_dataset = new_dataset["test"],
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <a name="Inference"></a>
    ### Inference
    Now let's try the model we just trained! First, let's first try the model without any GRPO trained:
    """
    )
    return


@app.cell
def _(SamplingParams, model):
    _text = 'What is the sqrt of 101?'
    _sampling_params = SamplingParams(temperature=1.0, top_k=50, max_tokens=1024)
    _output = model.fast_generate([_text], sampling_params=_sampling_params, lora_request=None)[0].outputs[0].text
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""And now with the LoRA we just trained with GRPO - we first save the LoRA first!""")
    return


@app.cell
def _(model):
    model.save_lora("grpo_saved_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Verify LoRA is actually trained!""")
    return


@app.cell
def _():
    from safetensors import safe_open

    tensors = {}
    with safe_open("grpo_saved_lora/adapter_model.safetensors", framework = "pt") as f:
        # Verify both A and B are non zero
        for key in f.keys():
            tensor = f.get_tensor(key)
            n_zeros = (tensor == 0).sum() / tensor.numel()
            assert(n_zeros.item() != tensor.numel())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now we load the LoRA and test:""")
    return


@app.cell
def _(SamplingParams, model, system_prompt, tokenizer):
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': 'What is the sqrt of 101?'}]
    _text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    _sampling_params = SamplingParams(temperature=1.0, top_k=50, max_tokens=2048)
    _output = model.fast_generate(_text, sampling_params=_sampling_params, lora_request=model.load_lora('grpo_saved_lora'))[0].outputs[0].text
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Our reasoning model is much better - it's not always correct, since we only trained it for an hour or so - it'll be better if we extend the sequence length and train for longer!""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
