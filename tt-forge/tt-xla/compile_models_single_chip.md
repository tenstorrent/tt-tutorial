# Compiling Models in PyTorch and JAX Using TT-XLA (Single Chip)

This tutorial explains how to write code to compile models for execution on Tenstorrent hardware using the TT-XLA frontend. TT-XLA is Tenstorrent's primary open-source compiler for machine learning that takes computation graphs from PyTorch and JAX frameworks, applies optimizations, and compiles them into optimized, low-level machine code for Tenstorrent chips.

The TT-XLA frontend supports: 

* **PyTorch** - A robust, flexible framework for building, training, and deploying neural networks and other machine learning models. 
* **JAX** - A Python framework that combines a NumPy-like API for array manipulation with composable function transformations for automatic differentiation, vectorization, and just-in-time (JIT) compilation using XLA for acceleration on Tenstorrent hardware, GPUs, and TPUs. 

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

# By default torch_xla uses the CPU device. The following must be run before acquiring/connecting to a device. 
xr.set_device_type("TT")

# Connect the device.
device = xm.xla_device() 

# Define compiler options. (You do not need to include all of these. See the chart below for further details.)
options = {
    "enable_optimizer": "true", 
    "enable_memory_layout_analysis": "true",
    "enable_l1_interleaved": "true",
    "enable_bfp8_conversion": "true",
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
| `xr.set_device_type("TT")` | Sets the target device to Tenstorrent hardware ("TT"). |
| `options = {...}` | Defines a dictionary of TT-XLA compiler flags.|
| `"enable_optimizer": "true"` | A TT-XLA compiler flag, it goes in `options = {...}`. Enables optimizer passes in MLIR. This includes various optimizations such as improving tensor memory layouts, operation configurations etc. |
| `"enable_memory_layout_analysis": "true"` | A TT-XLA compiler flag, it goes in `options = {...}`. Enables memory layout analysis to allow sharded memory layouts in optimizer passes. |
| `"enable_l1_interleaved": "true"` | A TT-XLA compiler flag, it goes in `options = {...}`. Enables L1 interleaved fallback analysis in optimizer passes. This analysis attempts to move tensors from DRAM to L1 memory with interleaved layout when beneficial for performance. |
| `"enable_bfp8_conversion": "true"`| Enables automatic MLIR graph conversion into block fp8 format. This is supported only when the graph is in bfloat16 format, to avoid loss in precision. The final graph has input and output nodes in bfloat16 and everything else in bfp8. Essentially adding type casts at the beginning and in the end of the graph, while all intermediate results are in bfp8. This bfloat16 wrapping is done because block formats are TT hardware specific, and the user should provide and get tensors of common dtype.|
| `"enable_fusing_conv2d_with_multiply_pattern": "true"` | A TT-XLA compiler flag, it goes in `options = {...}`. Enables Conv2d fusion with multiply pattern in the TTNN fusing pass.|
| `torch_xla.set_custom_compile_options(options)` | This command globally registers any compiler options you select. | 
| `model.compile(backend="tt")` | Equivalent to calling `torch.compile` on a `torch.nn.Module` "model" with the "tt" backend. |

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
| `serialize_compiled_artifacts_to_disk(model, input_data, output_prefix="out)` | Lowers your model into internal formats used by the compiler, collects internal compiler representations such as HLO, MLIR, or TT-IR, and saves these artifacts to disk. This is not required to run a model, but it gives you access to the compilation details, which can be useful for debugging, performance tuning, deployment, and reproducibility.|
| `jax.jit(model, backend="tt")` | The backend="tt" argument explictly tells JAX to use the TT-XLA compiler to generate code for Tenstorrent hardware. |





