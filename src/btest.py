import json
import os
import sys

import tqdm

import fire
import gradio as gr
import torch
import transformers
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "",
    test_file: str = "",
    prompt_template: str = "",
    server_name: str = "0.0.0.0",
    share_gradio: bool = False,
    batch_size: int = 8,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    # Initialize model
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate_batch(
        prompts,
        temperature=0.01,
        top_p=0.1,
        top_k=40,
        num_beams=1,
        max_new_tokens=528,
        **kwargs
    ):
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        decoded_outputs = [tokenizer.decode(seq) for seq in generation_output.sequences]
        return [prompter.get_response(output) for output in decoded_outputs]

    # Loading the test file
    file = json.load(open(test_file, "r"))
    try:
        responses = json.load(open('./responses.json','r'))
    except:
        responses = []
#     remove base on

    file = file[len(responses):]
    

    # Using batches for testing
    for i in tqdm.tqdm(range(0, len(file), batch_size)):
        batch = file[i : i + batch_size]
        instructions = [row["instruction_only_labels"] for row in batch]
        inputs = [row["text"] for row in batch]

        # Creating prompts
        prompts = [
            prompter.generate_prompt(inst, inp)
            for inst, inp in zip(instructions, inputs)
        ]

        # Evaluating batch
        batch_responses = evaluate_batch(prompts, top_p=1.0)
        for row, response in zip(batch, batch_responses):
            # Processing response, similar to the original code
            response = response.replace("</s>", "")
            try:
                responses.append({"item": row, "response": json.loads(response)})
            except:
                response = evaluate_batch(
                    [
                        prompter.generate_prompt(
                            row["instruction_only_labels"], row["text"]
                        )
                    ],
                    temperature=0.001,
                    top_p=0.9,
                )[0].replace("</s>", "")
                responses.append({"item": row, "response": json.loads(response)})
            # print()
        
        # every 10 batches, save the responses
        if i % 10 == 0:
            json.dump(responses, open("responses.json", "w"))

    # remaining batch
    if len(file) % batch_size != 0:
        batch = file[-(len(file) % batch_size) :]
        instructions = [row["instruction_only_labels"] for row in batch]
        inputs = [row["text"] for row in batch]

        # Creating prompts
        prompts = [
            prompter.generate_prompt(inst, inp)
            for inst, inp in zip(instructions, inputs)
        ]

        # Evaluating batch
        batch_responses = evaluate_batch(prompts, top_p=1.0)
        for row, response in zip(batch, batch_responses):
            # Processing response, similar to the original code
            response = response.replace("</s>", "")
            try:
                responses.append({"item": row, "response": json.loads(response)})
            except:
                response = evaluate_batch(
                    [
                        prompter.generate_prompt(
                            row["instruction_only_labels"], row["text"]
                        )
                    ],
                    temperature=0.001,
                    top_p=0.9,
                )[0].replace("</s>", "")
                responses.append({"item": row, "response": json.loads(response)})
            print()

    # save json
    with open("./test_labels.json", "w") as f:
        json.dump(responses, f)

    # ... (Gradio interface code)


if __name__ == "__main__":
    fire.Fire(main)
