# Convert Salesforce/codegen-350M-mono Code From HuggingFace For Use With Tenstorrent
CodeGen is a family of AI models that allow you to generate code from scratch or complete an existing function. Salesforce/codegen-350M-mono is one of the smallest available CodeGen models with 350 million parameters and a focus on Python. Unlike some of the larger CodeGen models, Salesforce/codegen-350M-mono is trained only on Python code. It is best used for tasks like creating simple Python code, function autocompletion, and code formatting.

In this walkthrough, you convert an existing HuggingFace code sample for Salesforce/codegen-350M-mono to be compatible with TT-Torch and Tenstorrent hardware. You then download the TT-Torch wheel and use it to compile and run the new sample code. If you are interested in using benchmarking features with your model, you must build from source to use TT-Torch tools and utilities. That is covered in a separate tutorial here - ADD TUTORIAL NAME. 

## Prerequisites

For this walkthrough you need:

* [Getting Started Instructions](getting_started.md) - Make sure you complete the following sections before continuing: 
    * [Configuring Hardware](getting_started.md#configuring-hardware)
    * [Installing a Wheel and Running an Example](getting_started.md#installing-a-wheel-and-running-an-example) - Steps 1-3, 5-6. (You do NOT need to install the packages listed in step 4 for the demo in the instructions.)

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

To convert this code for use with Tenstorrent, the main things that need to change are: 
* Addition of the Tenstorrent compiler.
* Conversion of models and tensors to be in `torch.bfloat16` format. 

## Salesforce/codegen-350M-mono Example Using Tenstorrent Specific Code



```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# https://huggingface.co/docs/transformers/v4.53.2/en/model_doc/codegen#usage-example
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tt_torch.dynamo.backend import BackendOptions, CompilerConfig

def codegen_demo():

    checkpoint = "Salesforce/codegen-350M-mono"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tokenizer.pad_token_id 

    text = "def hello_world():"
    inputs = tokenizer(text, return_tensors="pt")
    for k, v in inputs.items():
        if v.dtype == torch.float32:
            inputs[k] = v.to(torch.bfloat16)

    cc = CompilerConfig()
    # set any specific compiler options here
    # e.g., cc.dump_info = True

    # Initialize the backend options with the compiler configuration
    options = BackendOptions()
    options.compiler_config = cc

    # Compile the whole model
    model = torch.compile(model, backend="tt", options=options)

    # Setting max_new_tokens is optional
    completion = model.generate(**inputs, max_new_tokens=9, pad_token_id=model.config.pad_token_id)

    print(tokenizer.decode(completion[0]))
    

if __name__ == "__main__":
    codegen_demo()
```