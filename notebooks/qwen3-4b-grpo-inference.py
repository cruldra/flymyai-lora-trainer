import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Imports""")
    return


@app.cell
def _():
    # from unsloth import FastLanguageModel
    # import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    return AutoModelForCausalLM, AutoTokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Inference without Reasoning""")
    return


@app.cell
def _(AutoModelForCausalLM, AutoTokenizer):
    model_id = "unsloth/Qwen3-4B-Base"

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )

    prompt = "Which is the sqrt of 101?"
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.8,
        eos_token_id=tok.eos_token_id
    )

    print(tok.decode(out[0], skip_special_tokens=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<br><br><br><br>""")
    return


@app.cell
def _():
    from unsloth import FastLanguageModel
    import torch
    return (FastLanguageModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Inference with Reasoning""")
    return


@app.cell
def _(FastLanguageModel):
    max_seq_length = 2048  # Can increase for longer reasoning traces
    lora_rank = 32  # Larger rank = smarter, but slower
    model_1, tokenizer = FastLanguageModel.from_pretrained(model_name='unsloth/Qwen3-4B-Base', max_seq_length=max_seq_length, load_in_4bit=False, fast_inference=True, max_lora_rank=lora_rank, gpu_memory_utilization=0.7)
    model_1 = FastLanguageModel.get_peft_model(model_1, r=lora_rank, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], lora_alpha=lora_rank * 2, use_gradient_checkpointing='unsloth', random_state=3407)  # False for LoRA 16bit  # Enable vLLM fast inference  # Reduce if out of memory  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128  # *2 speeds up training  # Reduces memory usage
    return model_1, tokenizer


@app.cell
def _():
    reasoning_start = "<start_working_out>" # Acts as <think>
    reasoning_end   = "<end_working_out>"   # Acts as </think>
    solution_start  = "<SOLUTION>"
    solution_end    = "</SOLUTION>"

    system_prompt = \
    f"""You are given a problem.
    Think about the problem and provide your working out.
    Place it between {reasoning_start} and {reasoning_end}.
    Then, provide your solution between {solution_start}{solution_end}"""
    system_prompt
    return reasoning_start, system_prompt


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
    mo.md(r"""## Inference with Reasoning""")
    return


@app.cell
def _(model_1, system_prompt, tokenizer):
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': 'What is the sqrt of 101?'}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=True)
    from vllm import SamplingParams
    sampling_params = SamplingParams(temperature=1.0, top_k=50, max_tokens=2048)
    output = model_1.fast_generate(text, sampling_params=sampling_params, lora_request=model_1.load_lora('dont_touch/grpo_saved_lora'))[0].outputs[0].text
    output = f'<start_working_out>{output}'
    output  # Must add for generation
    return (output,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<br><br><br><br>""")
    return


@app.cell
def _(output):
    import pprint
    pprint.pprint(output)
    return


@app.cell
def _(output):
    output
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
