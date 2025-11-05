# Compiling JAX Models with TT-XLA for Multi-Chip Execution Using Tensor Parallelism
This tutorial shows 

If you want to try a simpler compilation process using a single chip and either JAX or PyTorch, please see:
* [Compiling Models in PyTorch and JAX Using TT-XLA (Single Chip)](#compile_models_single_chip.md)

If you want an example showing tensor parallelism instead of data parallelism, please see:
* [Compiling JAX Models with TT-XLA for Multi-Chip Execution Using Data Parallelism](#compile_models_multi_chip.md)

>**NOTE:** Training is not covered, this tutorial shows you how to do inference only. 

# System Configuration 

Before you get started with this tutorial, make sure you have:
* [Configured your Tenstorrent hardware.](https://github.com/tenstorrent/tt-xla/blob/main/docs/src/getting_started.md) 
* Set up your selected environment for working with TT-XLA. (For this walkthrough, you only need to [install the wheel](https://github.com/tenstorrent/tt-xla/blob/main/docs/src/getting_started.md#installing-a-wheel-and-running-an-example), however you could also choose to setup [Docker](https://github.com/tenstorrent/tt-xla/blob/main/docs/src/getting_started_docker.md) or [build from source](https://github.com/tenstorrent/tt-xla/blob/main/docs/src/getting_started_build_from_source.md). The quickest and simplest option is using the wheel.)

# Check Available Devices (Optional)

You can check how many Tenstorrent devices you have available by running the following code: 

```python
import jax

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
4. Wrap the model with `jit`.
5. Use tensor parallelism as the sharding strategy. (This step varies however, in this tutorial, tensor parallelism is shown.)

# Code Sample Showing a Single-Layer MLP and Tensor Parallelism

This code sample uses JAX and Flax to compile and execute a single-layer (dense layer) multi-layer perceptron (MLP) using multiple Tenstorrent chips. The weight matrix and input data are sharded across devices along the feature axis to achieve tensor parallelism. 

You can use this example as a template for your own code. 

The code teaches you: 
* How to write and compile a tensor-parallel JAX model using TT-XLA
* How to initialize parameters on CPU and then move them to devices
* How to create a `Mesh` and use `NamedSharding` for multi-chip execution
* How to do manual parameter and data sharding (useful for performance control, debugging and matching logical sharding to physical devices)

```python
import os
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

# Define a model with a single Dense layer, weights sharded over devices 
class ShardedDenseMLP(nn.Module):
    features: int = 8  # output features

    @nn.compact
    def __call__(self, x):
        # Shard the kernel weights column-wise (along output features dimension)
        dense = nn.Dense(
            self.features,
            param_dtype=jnp.float32,
            kernel_init=nn.linear.default_kernel_init,
            use_bias=False,
        )
        return dense(x)

def initialize_model_cpu(rng_key, input_shape):
    model = ShardedDenseMLP()
    with jax.default_device(jax.devices("cpu")[0]):
        dummy_input = jnp.zeros(input_shape, dtype=jnp.float32)
        params = model.init(rng_key, dummy_input)

    return model, params

def make_jit_forward(model, mesh):
    def forward(params, x):
        return model.apply(params, x)

    in_shardings = (
        {  # params sharding: shard the kernel matrix along axis 1 (features)
            'params': {
                'Dense_0': {
                    'kernel': NamedSharding(mesh, P(None, 'model'))
                }
            }
        },
        NamedSharding(mesh, P()),  # input features replicated
    )
    out_sharding = NamedSharding(mesh, P(None, "model"))
    return jax.jit(forward, in_shardings=in_shardings, out_shardings=out_sharding)

def main():
    rng = random.PRNGKey(0)

    # Input shape: (batch, features)
    batch_size = 8
    features = 8

    model, params = initialize_model_cpu(rng, (batch_size, features))

    # Create mesh over TT devices (at least 2 devices)
    tt_devices = jax.devices("tt")
    if len(tt_devices) < 2:
        raise RuntimeError("Need at least 2 Tenstorrent devices for this example")


    mesh_devices = tt_devices[:2]
    mesh = Mesh(mesh_devices, axis_names=("model",))

    # Initialize input data
    input_data = jnp.ones((batch_size, features), dtype=jnp.float32)

    # Shard params accordingly
    sharded_params = jax.tree.map(
        lambda x: jax.device_put(x, NamedSharding(mesh, P(None, "model"))) if x.ndim == 2
               else jax.device_put(x, NamedSharding(mesh, P())),
        params['params']
    )
    params = {"params": sharded_params}

    jit_forward = make_jit_forward(model, mesh)

    with mesh:
        output = jit_forward(params, input_data)

    print("Output:", output)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()
```

Elements of the code that are Tenstorrent-specific are described in the following chart: 

| Code | Explanation | 
|---|---| 
| `jax.devices("tt")` | Required to create a device mesh on Tenstorrent hardware only. | 
| `Mesh(mesh_devices, axis_names=("model",))` / `NamedSharding(mesh, P(None, "model"))`| Sharding over multiple Tenstorrent devices requires a `Mesh` object to tell JAX how to distribute computation and data across chips. | 
| `PartitionSpec(None, "model")` | Tenstorrent currently supports only certain sharding patterns, for example data parallelism using the batch dimension. For tensor parallelism, you can shard along the model or output dimension. | 
| `jax.device_put(...)` | Tenstorrent's backend does not currently support automatic parameter sharding, so you need to do it manually using `device_put`. | 
| `jax.default_device(jax.devices("cpu")[0])` | Tenstorrent devices don't support on-device random number generation, so to initialize the parameters with random values, you must use the CPU. | 

# Where to Go Next 

In this tutorial you learned how to define and compile a JAX model using TT-XLA and tensor parallelism for execution across multiple chips. The example is designed as a template for building more complex models with custom sharding strategies. 

If you are interested in learning more, you may find the following tutorials useful: 
* [Compiling JAX Models with TT-XLA for Multi-Chip Execution Using Data Parallelism](#compile_models_multi_chip.md)
* [Compiling Models in PyTorch and JAX Using TT-XLA (Single Chip)](#compile_models_single_chip.md)