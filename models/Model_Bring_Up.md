# Model Bring-Up

This Guide covers basic steps for model bring-up on TT Devices.

## Basic Requirements

- Access to TT-Hardware | [Buy TT-Hardware](https://tenstorrent.com/hardware/wormhole)
- Knowledge of PyTorch and transformers.
- Familiarity with [TT-Metalium](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/index.html) and [TT-NN](https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html).
- See: [TT-Metalium README.md](https://github.com/tenstorrent/tt-metal/blob/main/README.md) for the latest updates to Tenstorrent models.
- See: [TT-Metalium Tech Reports](https://github.com/tenstorrent/tt-metal?tab=readme-ov-file#tt-metalium-tech-reports) for information on TT-Metalium.

## Run a Demo

After setting up the environment correctly, run a demo to test the environment.

> [!TIP]
> Use the Llama3 codebase for transformer based models. Pick a model most similar to the model being brought up.

- Determine which model you are configuring. Here is a list of [Tenstorrent Models](https://github.com/tenstorrent/tt-metal?tab=readme-ov-file#llms).
- Model details are available in the TT-Metalium GitHub repository: [TT-Metalium Model Demos](https://github.com/tenstorrent/tt-metal/tree/main/models/demos).

## LLM Implementation

### Baseline Validation

- Select a reference model based on the model to be brought up.
- Follow standard instructions to run the reference model on CPU/GPU for baseline validation.
- Ensure the model is set up correctly with proper weights, attributes, etc. before adapting it for TT-Devices.

### Implementation

- Adapt the model codebase as reference implementation. Select a model that is the most similar to the model to be brought up, again Llama3 should be used with transformer based models:
  - Make a copy of the model codebase.
  - Modify modules with model dimensions as needed.
  - Use a single device first for simpler bring-up; Wormhole has 12 GB DRAM storage and can support models up to 12B parameters in BFP8. If possible, use smaller versions of the model that fit on a single device. The model can scaled up in size and on more devices from here.

> [!NOTE]
>  - The decode layer supports batch=32. Each row is a separate user in 32x32 tiles used by the TT-Metalium stack.
>  - In prefill, rows map to different input tokens. Implement prefill with batch=1; prefill is compute-bound and multiple batches do not benefit performance.
>  - See: [Converting Torch Model to TT-NN](https://docs.tenstorrent.com/docs-test/ttnn/latest/ttnn/converting_torch_model_to_ttnn.html) for model conversion.

## Systematic Component-wise Model Bring-Up

1. Bring-up decode stage modules first.
2. Bring-up each individual decode module separately.
   - Implement the module in TT-NN then pass the same inputs to the reference and the TT-NN modules to check for correctness.
   - Create a unit test with model dimensions, feed random data activations and real weights.
   - Verify that output PCC match the reference output, use reference implementation for validation.
   - Unit tests are useful for the analysis/accuracy analysis layer.
3. Test each module by verifying the PCC.
> [!NOTE]
> Examples of standard modules used are: Layernorm/RMSNorm, RotaryEmbedding, Attention, or MultiLayer Perceptron (MLP).
4. Compose all modules into higher level modules like single layer decoder or full decoder.
5. Implement decode mode then use decode to run prefill.
6. Test the model configuration without a dedicated prefill implementation.
7. Create a full model test. Use real inputs to produce real outputs; for LLMs, input text to output decode tokens.
8. Run the same inputs through the reference and TT-NN models to check the accuracy of your implementation. Teacher forcing is the ideal method to use with LLMs.
9. Generate tokens from the reference and TT-NN models. Input the reference tokens into both models in the next iteration. Depending on differences in the outputs, you can check accuracy metrics.
10. Verify the output tokens are:
    - Meaningful and coherent.
    - Similar to reference model tokens.
> [!NOTE]
> Due to differences in floating point arithmetic and non-linear approximations, tokens may not be exact matches.
11. Prefill Implementation:
    - Bring-up layer-by-layer similar to decode.
    - Run the entire model including prefill and decode.
See: [LLMs Bring-up in TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/LLMs/llms.md) or [ViT in TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/LLMs/llms.md) for more information on these steps.

## Model Performance Optimization

Optimization tools like Metal Trace, async mode, and multiple command queues improve model performance. See [Advanced Performance Optimizations for Models](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md#1-metal-trace) for information on performance optimization.

## Data Parallel Implementation

Determine how many copies of a model can be run by dividing the model size by the available memory on device. For example:

- Wormhole n150 has 12GB of storage supporting models up to 12B parameters in BFP8.
- The Llama 3.1 model size is 8B, each Wormhole n150 can run a copy of it.
- A TT-LoudBox (TW-02001) has four Wormhole n150s. Using data parallel scaling, it can run four independent instances of Llama 3.1 to increase throughput.
- Large models like Falcon 40B do not fit on a single device. At least two Wormhole n300s (24GB each) are required to run in tensor parallel scaling where single operations are distributed across devices.
- A TT-QuietBox System has four Wormhole n300s; it can run two copies of Falcon 40B with each copy running on two Wormhole n300 cards.
- Requirements:
  - Weights must be replicated on different devices.
  - Different inputs must be sent to different devices.
  - Can be done using the device mesh APIs in TT-NN.
  - We recommend adding data parallel support to each module separately and unit test each module before running the entire model.
- See: [Multi-Device Reference](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/LLMs/llms.md#33-multi-device) for information on data parallel implementation.
