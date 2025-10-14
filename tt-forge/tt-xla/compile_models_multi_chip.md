# Compiling JAX Models with TT-XLA for Multi-Chip Execution Using Data Parallelism

This tutorial explains how to write code to compile models for execution on Tenstorrent hardware using JAX, the TT-XLA frontend, data parallelism, and multiple Tenstorrent chips. 

To set up multi-chip execution, you need to:
* Define a mesh of devices
* Explain how to split the work
* Use tools like `shard_map` and `PartitionSpec` to control data and computation sharding
* Use collective communication operations across chips 

If you want to try a simpler compilation process using a single chip and either JAX or PyTorch, please see:
* [Compiling Models in PyTorch and JAX Using TT-XLA (Single Chip)](#compile_models_single_chip.md)

If you want an example showing tensor parallelism instead of data parallelism, please see:
* [Compiling JAX Models with TT-XLA for Multi-Chip Execution Using Tensor Parallelism](#compile_multi_chip_w_tensor.md)

>**NOTE:** Training is not covered, this tutorial shows you how to do inference only. 

# System Configuration 

Before you get started with this tutorial, make sure you have:
* [Configured your Tenstorrent hardware.](https://github.com/tenstorrent/tt-xla/blob/main/docs/src/getting_started.md) 
* Set up your selected environment for working with TT-XLA. (For this walkthrough, you only need to [install the wheel](https://github.com/tenstorrent/tt-xla/blob/main/docs/src/getting_started.md#installing-a-wheel-and-running-an-example), however you could also choose to setup [Docker](https://github.com/tenstorrent/tt-xla/blob/main/docs/src/getting_started_docker.md) or [build from source](https://github.com/tenstorrent/tt-xla/blob/main/docs/src/getting_started_build_from_source.md). The quickest and simplest option is using the wheel.)

# Check Available Devices (Optional)

You can check how many Tenstorrent devices you have available by running the following code: 

```python
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import pjit

# Query available devices
devices = jax.devices()
print("Available devices:")
for i, device in enumerate(devices):
    print(f"  [{i}] {device}")

# Optional: check device type
print(f"JAX backend platform: {jax.default_backend()}")
```

In the output, you should see something like this that shows how many Tenstorrent devices you have and what type they are: 

```python
Available devices:
  [0] TTDevice(id=0, arch=Wormhole_b0)
  [1] TTDevice(id=1, arch=Wormhole_b0)
JAX backend platform: tt
```

# Overview for Compiling a JAX Model with Multi-Chip Execution

When compiling a model for multi-chip execution, the general steps are:

1. Import JAX and retrieve available device information. 
2. Setup a logical mesh across your multi-chip system.
3. Define a simple model (or load an existing one).
4. Wrap the model with `pjit`. (`jit` is used when you are running on a single device.)
5. Use data parallelism as the sharding strategy. (This step varies however, in this tutorial, data parallelism is shown.)

# Code Sample Showing Minimalist MLP Inference and Data Parallelism

This code sample creates a minimalist multi-layer perceptron (MLP) inference example with Flax that you initialize on CPU, then execute across two Tenstorrent chips using `pjit`. The model includes two dense layers with a ReLU activation in between. The input batch is sharded over a 1D data mesh, while parameters are replicated across devices.

The code teaches you:
* How to write a model in Flax (`nn.Module`)
* How to initialize on CPU to avoid TT errors
* How to move data and parameters to TT devices 
* How to use `pjit` and `Mesh` to shard inputs across TT chips 
* How to run a compiled inference pass using TT 

You can use this sample as a template for creating your own code. 

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental.pjit import pjit
import os

# Optional: Force TT backend
os.environ["JAX_PLATFORMS"] = "tt"

# ---------------------------
# 1. Simple MLP Model
# ---------------------------
class SimpleMLP(nn.Module):
    features: int = 8

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(self.features)(x)
        return x

# ---------------------------
# 2. Initialize Model on CPU
# ---------------------------
def initialize_model_cpu(rng_key, input_shape):
    model = SimpleMLP()
    with jax.default_device(jax.devices("cpu")[0]):
        dummy_input = jnp.zeros(input_shape, dtype=jnp.float32)
        params = model.init(rng_key, dummy_input)
    return model, params

# ---------------------------
# 3. pjit-wrapped forward pass
# ---------------------------
def make_pjit_forward(model):

    def forward(params, x):
        return model.apply(params, x)

    return pjit(
        forward,
        in_shardings=(None, P("data")),   # No param sharding, batch sharded
        out_shardings=P("data")
    )

# ---------------------------
# 4. Main
# ---------------------------
def main():
    rng = random.PRNGKey(0)
    input_shape = (8, 8)  # batch = 8 (will shard across 2 chips)

    # Initialize model on CPU
    model, params = initialize_model_cpu(rng, input_shape)

    # Prepare data
    input_data = jnp.ones(input_shape, dtype=jnp.float32)

    # Create 1D mesh over TT devices (2 chips)
    tt_devices = jax.devices("tt")
    if len(tt_devices) < 2:
        raise RuntimeError("Need at least 2 TT devices for this example.")

    mesh = Mesh(tt_devices[:2], axis_names=("data",))

    # Shard input
    sharding = NamedSharding(mesh, P("data"))
    input_data = jax.device_put(input_data, sharding)

    # pjit forward
    pjit_forward = make_pjit_forward(model)

    with mesh:
        output = pjit_forward(params, input_data)

    print("Output:", output)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()
```

Elements of the code that are Tenstorrent-specific are described in the following chart: 

| Code | Explanation |
|---|---|
| `os.environ["JAX_PLATFORMS"] = "tt"` | This forces JAX to compile and run the model using the Tenstorrent backend (`tt`). TT-XLA registers itself as a JAX backend named `tt`. If you do not include this, JAX defaults to CPU or GPU and does not use Tenstorrent. |
| `tt_devices = jax.devices("tt")` | Using this with `tt` retrieves information about what Tenstorrent devices are available. |
| `with jax.default_device(jax.devices("cpu")[0]) ... params = model.init(rng_key, dummy_input)` | While this snippet is not Tenstorrent-specific, the way it is used is. Tenstorrent does not currently support random number generation on device, so this code shows how to have parameter initialization happen on the CPU. It can then be moved to Tenstorrent for execution. |
| ```mesh = Mesh(tt_devices[:2], axis_names=("data",)) pjit_forward = pjit(..., in_shardings=(None, P("data")), ...)``` | `pjit` is a standard JAX tool. This code snippet is called out because Tenstorrent's backend has limited sharding support. Primarily, data parallelism is supported. | 

>**NOTE:** JAX and TT-XLA is currently the best way to run multi-chip workloads on Tenstorrent hardware. 

In this tutorial you learned how to define and compile a JAX model using TT-XLA and data parallelism for execution across multiple chips. The example is designed as a template for building your own models.

If you are interested in learning more, you may find the following tutorials useful: 
* [Compiling JAX Models with TT-XLA for Multi-Chip Execution Using Tensor Parallelism](#compile_multi_chip_w_tensor.md)
* [Compiling Models in PyTorch and JAX Using TT-XLA (Single Chip)](#compile_models_single_chip.md)