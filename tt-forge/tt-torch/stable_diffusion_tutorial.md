# Convert Stable Diffusion Code From HuggingFace For Use With Tenstorrent
Stable Diffusion

## Prerequisites 

For this walkthrough you need:

* [Getting Started Instructions](getting_started.md) - Make sure you complete the following sections before continuing: 
    * [Configuring Hardware](getting_started.md#configuring-hardware)
    * [Installing a Wheel and Running an Example](getting_started.md#installing-a-wheel-and-running-an-example) - Steps 1-3, 5-6. (You do NOT need to install the packages listed in step 4 for the demo in the getting started instructions.)
* diffusers Python package
* Hugging Face account

## Authenticating With HuggingFace 

This section explains how to use HuggingFace to access the model repository and authenticate with HuggingFace so you can run the Stable Diffusion model. To use the Stable Diffusion model, do the following: 

1. Log in to your HuggingFace account.

2. Navigate to the **[stable-diffusion.3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)** page. Follow the instructions there to obtain access to the model. 

3. You also need a token to authenticate the repository. To do that, click your **profile image** in the upper right corner of the screen. 

4. From the drop-down menu, click **Settings**. 

5. On the Settings screen, from the menu on the left, click **Access Tokens**. 

6. Click the **Create new token** button. 

7. Name your token. For this walkthrough, the token name **diffusion-token** is used. 

8. For the scope of the token, because you are just running a model, a **Read** token is fine. Click **Read** at the top of the Create new Access Token screen. 

9. Click **Create token**. 

10. You are prompted to save your token. Keep it somewhere where you can access it easily since you will need to add it to the code in the next section. Click **Done**.

## Convert HuggingFace Stable Diffusion stable-diffusion.3.5-medium Example to Work With Tenstorrent

This section provides the original code from the [HuggingFace for **stable-diffusion.3.5-medium**](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_3#usage-example) code example and shows you what is required to convert it to work with TT-Torch. 

```python
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(
    prompt="a photo of a cat holding a sign that says hello world",
    negative_prompt="",
    num_inference_steps=28,
    height=1024,
    width=1024,
    guidance_scale=7.0,
).images[0]

image.save("sd3_hello_world.png")
```

The sample loads the Stable Diffusion 3 Medium model from HuggingFace and sets the model to use 16-bit floating point precision to save memory and improve speed (on supported GPUs). It transfers the model to a CUDA-enabled GPU if available, then sets the requirements for the desired image. Image parameters for this example include: 

* prompt - A description of the image to generate.
* negative_prompt - Anything you do not want included in the generated image. 
* num_inference_steps - How many denoising steps you want. More steps yield higher quality, but take more time. 
* height / width - The resolution of the generated image. 
* guidance_scale - The scale ranges from 5.0-10.0, and providing a higher number means the output image more closely follows the prompt. 

## stable-diffusion.3.5-medium Example Using Tenstorrent Specific Code 

This section takes the code sample from the last section and shows you what changes need to be made for use with Tenstorrent hardware and the TT-Torch frontend. The key changes are: 

* `cc = CompilerConfig()` - This is for TT-Torch's compiler. Learn more about the available options in [utils.py](https://github.com/tenstorrent/tt-torch/blob/main/tt_torch/tools/utils.py) 
* `BackendOptions()` - This lets you choose different options you may want to add when using the TT-Torch compiler, for example if you want to enable async execution or manually cleanup runtime tensors. Learn more about the available options in [backend.py](https://github.com/tenstorrent/tt-torch/blob/main/tt_torch/dynamo/backend.py)
* Adding a token so you can access the gated model. 

Here is the code: 

```python
import torch
import time
import os
from diffusers import StableDiffusion3Pipeline
from tt_torch.tools.utils import CompilerConfig
from tt_torch.dynamo.backend import BackendOptions

def configure_tenstorrent_backend():
    cc = CompilerConfig() 
    options = BackendOptions() # You can add additional options as needed
    options.compiler_config = cc
    return options

def stable_diffusion_demo():
    # Load pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        token=os.getenv("HF_TOKEN"),
        torch_dtype=torch.float32,
        text_encoder_3=None,
        tokenizer_3=None,
        low_cpu_mem_usage=True,
    )

    # Tokenizer handling
    if hasattr(pipe, "tokenizer_2"):
        pipe.tokenizer_2.truncation = True
        pipe.tokenizer_2.padding = True

    # Reduce memory usage
    pipe.enable_attention_slicing()

    # Only compile the UNet — the heaviest part
    try:
        pipe.transformer.forward = torch.compile(
            pipe.transformer.forward,
            backend="tt",
            options=configure_tenstorrent_backend()
        )
    except Exception as e:
        print(f"[ERROR] Compilation failed: {e}")
        return

    # Image generation
    torch.manual_seed(0)
    start = time.time()

    try:
        image = pipe(
            prompt="a photo of a cat holding a sign that says hello world",
            num_inference_steps=30,
            height=192,
            width=192,
            guidance_scale=6.5,
        ).images[0]
    except Exception as e:
        print(f"[ERROR] Image generation failed: {e}")
        return

    end = time.time()
    print(f"[INFO] Time taken: {end - start:.2f} seconds")

    image.save("sd3_hello_world.png")
    print("[INFO] Image saved as sd3_hello_world.png")

if __name__ == "__main__":
    stable_diffusion_demo()
```

An easy way to run this code is to do the following: 

1. The model is gated on HuggingFace, so you must authenticate with a HuggingFace token. The best practice is not to hard code the token into your code. Run the following command in your terminal, replacing `your_token_here` with your token: 

```bash
export HF_TOKEN=your_token_here
```

2. In your version of the **tt-forge repo**, navigate to **tt-forge/demos/tt-torch**. 

3. Use **nano** or another code editor to create a file, for example **hug_diff.py**. 

4. Save your file. 

5. Navigate back to the root **tt-forge**. 

6. Run: 

```python
demos/tt-torch/hug_codegen.py
```

7. If everything works, the model should generate a file called **sd3_hello_world.png**. The terminal outputs the following: 

```bash
[INFO] Time taken: 2709.52 seconds
[INFO] Image saved as sd3_hello_world.png
```

The first line tells you how long it took to run the model, and the second line tells you the name of the file your new image is saved as. 

## Additional Resources
If you want to learn more about working with TT-Torch, see the following resources:
* [TT-Torch Docs Pages](https://docs.tenstorrent.com/tt-torch/)
* [TT-Forge Repo with Demos for TT-Torch](https://github.com/tenstorrent/tt-forge/tree/main/demos/tt-torch)