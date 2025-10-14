# Convert Salesforce/codegen-350M-mono Code From HuggingFace For Use With Tenstorrent
CodeGen is a family of AI models that allow you to generate code from scratch or complete an existing function. Salesforce/codegen-350M-mono is one of the smallest available CodeGen models with 350 million parameters and a focus on Python. Unlike some of the larger CodeGen models, Salesforce/codegen-350M-mono is trained only on Python code. It is best used for tasks like creating simple Python code, function autocompletion, and code formatting.

In this walkthrough, you convert an existing HuggingFace code sample for Salesforce/codegen-350M-mono to be compatible with TT-Torch and Tenstorrent hardware. You then download the TT-Torch wheel and use it to compile and run the new sample code. 

## Prerequisites

For this walkthrough you need:

* [Getting Started Instructions](getting_started.md) - Make sure you complete the following sections before continuing: 
    * [Configuring Hardware](getting_started.md#configuring-hardware)
    * [Installing a Wheel and Running an Example](getting_started.md#installing-a-wheel-and-running-an-example) - Steps 1-3, 5-6. (You do NOT need to install the packages listed in step 4 for the demo in the getting started instructions.)

At the end of this section, you should have the TT-Torch wheel installed, a running virtual environment, and be inside the TT-Forge repo.

## Convert HuggingFace Salesforce/codegen-350M-mono Example to Work With Tenstorrent

This section provides the original code from the [HuggingFace for **Salesforce/codegen-350M-mono**](https://huggingface.co/docs/transformers/v4.53.2/en/model_doc/codegen#usage-example) code example and shows you what is required to convert it to work with TT-Torch.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "Salesforce/codegen-350M-mono"
model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

text = "def hello_world():"

completion = model.generate(**tokenizer(text, return_tensors="pt"))

print(tokenizer.decode(completion[0]))
```

The sample loads the saved version of the Salesforce/codegen-350M-mono model with associated model weights and configuration, sets up a tokenizer, and creates an example input string for the model to complete, in this case `def hello_world()`. The model then generates and prints out the final outcome in the terminal: 

```python
def hello_world():
    print("Hello, world!")
```

In the next section, the code is converted for use with Tenstorrent and TT-Torch.

## Salesforce/codegen-350M-mono Example Using Tenstorrent Specific Code

This section takes the code sample from the last section and shows you what changes need to be made for use with Tenstorrent hardware and the TT-Torch frontend. The key changes are: 

* `tokenizer.pad_token_id` is set - This ensures the input is all of the same fixed length. Without this change, every time input of a different length is encountered, it triggers a recompile. When the max number of recompilations is reached, the compiler will give up on compiling the function, resulting in a warning message. The model still runs, but in eager mode, which is much slower. 
* `torch.bfloat16` - It offers the same dynamic range as `float32`, with much faster performance on Tenstorrent hardware and lower bandwidth and memory usage. 
* `torch_xla.set_custom_compile_options({...})` - For details about the compiler, refer to [compiler_config.py](https://github.com/tenstorrent/tt-xla/blob/main/tests/infra/testers/compiler_config.py)

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# https://huggingface.co/docs/transformers/v4.53.2/en/model_doc/codegen#usage-example

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

def codegen_demo():
    checkpoint = "Salesforce/codegen-350M-mono"

    # Set up XLA runtime for TT backend
    xr.set_device_type("TT")

    # Optional: Configure codegen options
    torch_xla.set_custom_compile_options({
        "backend": "codegen_py",
        "export_path": "codegen_codegen",  # Adjust path as needed
    })

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Move model to XLA device
    device = xm.xla_device()
    model = model.to(device)
    model.eval()

    # Prepare input
    text = "def hello_world():"
    inputs = tokenizer(text, return_tensors="pt")
    for k, v in inputs.items():
        if v.dtype == torch.float32:
            inputs[k] = v.to(torch.bfloat16)
        inputs[k] = inputs[k].to(device)

    # Run inference (this will trigger compilation/codegen)
    completion = model.generate(
        **inputs,
        max_new_tokens=9,
        pad_token_id=model.config.pad_token_id,
    )

    print(tokenizer.decode(completion[0]))

if __name__ == "__main__":
    codegen_demo()

```

An easy way to run this code is to do the following: 

1. In your version of the **tt-forge repo**, navigate to **tt-forge/demos/tt-torch**. 

2. Use **nano** or another code editor to create a file, for example **hug_codegen.py**. 

3. Save your file. 

4. Navigate back to the root **tt-forge**. 

5. Run: 

```python
demos/tt-torch/hug_codegen.py
```

6. If everything works, the model should complete the function for `def hello_world()`, resulting in the following output: 

```python
def hello_world():
    print("Hello, world!")
```

## Additional Resources
If you want to learn more about working with TT-Torch, see the following resources:
* [TT-Torch Docs Pages](https://docs.tenstorrent.com/tt-torch/)
* [TT-Forge Repo with Demos for TT-Torch](https://github.com/tenstorrent/tt-forge/tree/main/demos/tt-torch)