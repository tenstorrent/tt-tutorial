# Compiling Models in PyTorch and JAX Using TT-XLA (Single Chip)

This tutorial explains how to write code to compile models for execution on Tenstorrent hardware using the TT-XLA frontend. TT-XLA is Tenstorrent's primary open-source compiler for machine learning that takes computation graphs from PyTorch and JAX frameworks, applies optimizations, and compiles them into optimized, low-level machine code for Tenstorrent chips.

The TT-XLA frontend supports: 

* **PyTorch** - A robust, flexible framework for building, training, and deploying neural networks and other machine learning models. 
* **JAX** - A Python framework that combines a NumPy-like API for array manipulation with composable function transformations for automatic differentiation, vectorization, and just-in-time (JIT) compilation using XLA for acceleration on GPUs and TPUs. 

# Choosing Between JAX and PyTorch 

| | PyTorch | JAX |
|--|--|--|
| Programming Paradigm | Object-oriented with dynamic graphs | Functional with static graphs after JIT compilation |
| Automatic Differentiation | Uses a backward() method on the loss | Uses function transformations like grad to return gradient functions|
| State Management | Manages state within objects | Requires explicit state passing |
| Performance Optimization | Relies on highly optimized C++ backends and a mature ecosystem for performance | Leverages JIT compilation and XLA for potentially higher performance, especially on TPUs | 
| Ecosystem and Community | More established, broaded ecosystem with more pre-built solutions | Growing ecosystem, often used for cutting-edge research | 

Both are good choices, PyTorch is more beginner friendly as it's more established, allows for rapid prototyping, and offers many pre-built solutions. JAX is best for maximum performance, fine-grained control, and its functional programming style for complex numerical computations and machine learning research. 

# System Configuration 

Before you get started with this tutorial, make sure you have:
* [Configured your Tenstorrent hardware.](https://github.com/tenstorrent/tt-xla/blob/main/docs/src/getting_started.md) 
* Set up your selected environment for working with TT-XLA. (For this walkthrough, you only need to [install the wheel](https://github.com/tenstorrent/tt-xla/blob/main/docs/src/getting_started.md#installing-a-wheel-and-running-an-example), however you could also choose to setup [Docker](https://github.com/tenstorrent/tt-xla/blob/main/docs/src/getting_started_docker.md) or [build from source](https://github.com/tenstorrent/tt-xla/blob/main/docs/src/getting_started_build_from_source.md). The quickest and simplest option is using the wheel.)

# Compiling a Model for Use with TT-XLA and PyTorch

When writing code for compiling a model for use with TT-XLA, if you are using PyTorch, you can use the [Torch-XLA APIs](https://docs.pytorch.org/xla/release/r2.8/learn/pytorch-on-xla-devices.html) and Tenstorrent extensions. Tenstorrent uses Torch-XLA to emit XLA HLO and then TT-XLA compiles it. 

The general structure is: 

```python
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

# By default torch_xla uses the CPU device so set it to the TT device.
xr.set_device_type("TT")

# Connect the device.
device = xm.xla_device() 

# Define compiler options. (You do not need to include all of these.)
options = {
    "enable_optimizer": "true",
    "enable_memory_layout_analysis": "true",
    "enable_l1_interleaved": "true",
    "enable_fusing_conv2d_with_multiply_pattern": "true",
}

# Register the compile options globally.
torch_xla.set_custom_compile_options(options)

# Put the model in inference mode if running inference. (This step is not necessary, but helps when you might encounter training errors.)
model = model.eval()

# Optional - Use the Tenstorrent backend to compile before running on device. The backend comes from tt-xla/blob/main/python_package/tt_torch/backend/backend.py
model.compile(backend="tt")

# Move inputs and model to device.
input = input.to(device)
model = model.to(device)

# Run the model.
output = model(input)

# Move output to CPU. 
output = output.to("cpu")
```

The following chart lists the aspects of the template that are Tenstorrent specific: 

| Code | Explanation |
|------|-------------|
| import torch_xla.runtime as xr | Imports the Tenstorrent specific runtime module, required to work with Tenstorrent devices. |
| xr.set_device_type("TT") | Sets the target device to Tenstorrent hardware ("TT"). |
| options = {...} | Defines a dictionary of TT-XLA compiler flags.|
| "enable_optimizer": "true" | A TT-XLA compiler flag, it goes in `options = {...}`. Enables TT-XLA compiler optimizations on the computation graph. |
| "enable_memory_layout_analysis": "true" | A TT-XLA compiler flag, it goes in `options = {...}`. It
allows for memory layout tuning. |
| "enable_l1_interleaved": "true" | A TT-XLA compiler flag, it goes in `options = {...}`. It boosts memory access efficiency across cores. |
| "enable_fusing_conv2d_with_multiply_pattern": "true" | A TT-XLA compiler flag, it goes in `options = {...}`. Enables fusion of Conv2D followed by Multiply into a single operation to reduce op count and memory movement. |
| torch_xla.set_custom_compile_options(options) | This command globally registers any compiler options you select| 
| model.compile(backend="tt") | A Tenstorrent specific method for compiling models. |

# Compiling a Model for Use with TT-XLA and JAX
This section provides a simple template for writing code for compiling a model using JAX and TT-XLA. Tenstorrent uses XLA natively to emit XLA HLO, and TT-XLA compiles the HLO just like with PyTorch. 

```python
import jax
import jax.numpy as jnp
from tt_jax import serialize_compiled_artifacts_to_disk  # Optional utility

# Define a simple JAX function.
def model(x):
    return jnp.sin(x) * jnp.cos(x)

# Create example input data.
input_data = jnp.ones((10,))

# Compile the function for the Tenstorrent backend.
# The `backend="tt"` argument explicitly selects TT-XLA.
compiled_model = jax.jit(model, backend="tt")

# Run the compiled model.
output = compiled_model(input_data)

print("Output:", output)

# Optional: serialize the compiled artifacts for deployment or inspection.
serialize_compiled_artifacts_to_disk(model, input_data, output_prefix="output/model")
```

The following chart lists the aspects of the template that are Tenstorrent specific: 

| Code | Explanation |
|---|---|
| from tt_jax import serialize_compiled_artifacts_to_disk | Imports a function from Tenstorrent's tt_jax module that takes your JAX model and compiles it for Tenstorrent hardware. |
| serialize_compiled_artifacts_to_disk(model, input_data, output_prefix="out) | Lowers your model into internal formats used by the compiler, collects internal compiler representations such as HLO, MLIR, or TT-IR, and saves these artifacts to disk. This is not required to run a model, but it gives you access to the compilation details, which can be useful for debugging, performance tuning, deployment, and reproducibility.|
| jax.jit(model, backend="tt") | The backend="tt" argument explictly tells JAX to use the TT-XLA compiler to generate code for Tenstorrent hardware. |





