# Compiling Multi-chip Models in JAX Using TT-XLA

This tutorial explains how to write code to compile models for execution on Tenstorrent hardware using JAX, the TT-XLA frontend, and multiple Tenstorrent chips. Multi-chip requires you to:
* Define a mesh of devices
* Explain how to split the work
* Use tools like `shard_map` and `PartitionSpec` to control data and computation sharding
* Use collective communication operations across chips 

If you want to try a simpler compilation process using a single chip and either JAX or PyTorch, please see:
* [Compiling Models in PyTorch and JAX Using TT-XLA (Single Chip)](#compile_models_single_chip.md)

# System Configuration 

Before you get started with this tutorial, make sure you have:
* [Configured your Tenstorrent hardware.](https://github.com/tenstorrent/tt-xla/blob/main/docs/src/getting_started.md) 
* Set up your selected environment for working with TT-XLA. (For this walkthrough, you only need to [install the wheel](https://github.com/tenstorrent/tt-xla/blob/main/docs/src/getting_started.md#installing-a-wheel-and-running-an-example), however you could also choose to setup [Docker](https://github.com/tenstorrent/tt-xla/blob/main/docs/src/getting_started_docker.md) or [build from source](https://github.com/tenstorrent/tt-xla/blob/main/docs/src/getting_started_build_from_source.md). The quickest and simplest option is using the wheel.)

