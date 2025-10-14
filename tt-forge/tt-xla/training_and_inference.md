
# Training a Model Workflow with a Single Tenstorrent Device

This tutorial walks through an example workflow for training models and running inference. Because Tenstorrent does not currently offer some of the ops you need to train on-chip, the recommended approach is to train on CPU, save the export parameters, then put the model on Tenstorrent for inference.

# System Configuration 

Before you get started with this tutorial, make sure you have:
* [Configured your Tenstorrent hardware.](https://github.com/tenstorrent/tt-xla/blob/main/docs/src/getting_started.md) 
* Set up your selected environment for working with TT-XLA. (For this walkthrough, you only need to [install the wheel](https://github.com/tenstorrent/tt-xla/blob/main/docs/src/getting_started.md#installing-a-wheel-and-running-an-example), however you could also choose to setup [Docker](https://github.com/tenstorrent/tt-xla/blob/main/docs/src/getting_started_docker.md) or [build from source](https://github.com/tenstorrent/tt-xla/blob/main/docs/src/getting_started_build_from_source.md). The quickest and simplest option is using the wheel.)

# Sample Code Showing Inference Workflow

This section provides a simple template for setting up to train a model, then run inference on a Tenstorrent device. 

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import flax.serialization

# 1. Define a simple model
class SimpleMLP(nn.Module):
    features: int = 8

    @nn.compact
    def __call__(self, x):
        const_init = nn.initializers.constant(0.01)
        x = nn.Dense(self.features, kernel_init=const_init, bias_init=const_init)(x)
        x = nn.relu(x)
        x = nn.Dense(1, kernel_init=const_init, bias_init=const_init)(x)
        return x

# 2. Create training state (params + optimizer)
def create_train_state(rng, model, learning_rate, input_shape):
    params = model.init(rng, jnp.ones(input_shape))  # Initialize parameters
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# 3. Define a loss function and train step
def mse_loss(params, apply_fn, x, y):
    preds = apply_fn(params, x)
    return jnp.mean((preds - y) ** 2)

@jax.jit
def train_step(state, x, y):
    def loss_fn(params):
        return mse_loss(params, state.apply_fn, x, y)
    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

# 4. Training loop
def train_model():
    rng = jax.random.PRNGKey(0)
    model = SimpleMLP()

    input_shape = (16, 4)  # batch size 16, feature dim 4
    state = create_train_state(rng, model, learning_rate=0.001, input_shape=input_shape)

    # Dummy data: y = sum of features (just an example)
    x_train = jnp.ones(input_shape)
    y_train = jnp.sum(x_train, axis=1, keepdims=True)

    for step in range(100):
        state = train_step(state, x_train, y_train)
        if step % 20 == 0:
            loss_val = mse_loss(state.params, state.apply_fn, x_train, y_train)
            print(f"Step {step}, Loss: {loss_val:.4f}")

    return model, state.params

# 5. Save parameters to disk
def save_params(params, path="model_params.msgpack"):
    params_bytes = flax.serialization.to_bytes(params)
    with open(path, "wb") as f:
        f.write(params_bytes)
    print(f"Parameters saved to {path}")

# 6. Load parameters from disk
def load_params(model, path="model_params.msgpack"):
    with open(path, "rb") as f:
        params_bytes = f.read()
    dummy_params = model.init(jax.random.PRNGKey(0), jnp.ones((1,4)))
    params = flax.serialization.from_bytes(dummy_params, params_bytes)
    print(f"Parameters loaded from {path}")
    return params

# --- Inference with TT-XLA ---

def get_tt_device():
    devices = jax.devices("tt")
    if not devices:
        raise RuntimeError("No tt-xla devices found. Make sure tt-xla backend is configured.")
    return devices[0]

def create_tt_inference_fn(model):

    tt_jit = partial(jax.jit, backend="tt")

    @tt_jit
    def tt_inference(params, x):
        return model.apply(params, x)

    return tt_inference

def run_inference_tt(model, params):
    tt_device = get_tt_device()
    x = jnp.array([[1., 2., 3., 4.]])
    x_tt = jax.device_put(x, device=tt_device)
    params_tt = jax.device_put(params, device=tt_device)

    tt_inference = create_tt_inference_fn(model)

    preds = tt_inference(params_tt, x_tt)
    print(f"Input (on tt-xla device): {x_tt}")
    print(f"Prediction (on tt-xla device): {preds}")

# --- End Tenstorrent-specific code ---

def main():
    model, trained_params = train_model()
    save_params(trained_params)

    # Later / elsewhere: load params and run inference on tt-xla device
    loaded_params = load_params(model)
    run_inference_tt(model, loaded_params)

if __name__ == "__main__":
    main()

```

Elements of the code that are Tenstorrent-specific are described in the following chart: 

| Code | Explanation |
|---|---|
| `train_model()` | Runs fully on CPU/GPU to avoid Tenstorrent issues with unsupported ops. | 
| `save_params()` | Saves trained model weights from CPU/GPU training to disk. This is necessary for moving the information to a Tenstorrent device for inference later on. | 
| `load_params()` | Used to load saved model weights into memory from a file like `.msgpack`. Necessary for running inference on a Tenstorrent device.|
| `nn.initializers.constant` | The Tenstorrent compiler is static and requires that random initializers use compile-time constants. Using a constant initializer avoids triggering unsupported dynamic random ops. | 
| `run_inference_tt()` | Runs inference using the Tenstorrent compiler with JAX. |
| `jax.device_put(..., device=...)` | Moves input and model weights onto the Tenstorrent hardware for execution. | 
| `@jax.jit` | Used to trigger the Tenstorrent compiler to compiler the inference function for hardware execution. |

# Where to Go Next 

In this tutorial you learned the workflow for training a model and transferring it to a Tenstorrent device for inference. 

If you are interested in learning more, you may find the following tutorials useful: 
* [Compiling JAX Models with TT-XLA for Multi-Chip Execution Using Data Parallelism](#compile_models_multi_chip.md)
* [Compiling JAX Models with TT-XLA for Multi-Chip Execution Using Tensor Parallelism](#compile_multi_chip_w_tensor.md)
* [Compiling Models in PyTorch and JAX Using TT-XLA (Single Chip)](#compile_models_single_chip.md)