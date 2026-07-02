## The build plan

I want to create a more memory safe and rigorous implementation of the marian-fork that can run the specially quantized Firefox models with `int8shiftAlphaAll` mode. We have a custom fork of the Marian translation system to run CPU-optimized models.

In the original inference engine there is the [ExpressionGraph](inference/marian-fork/src/graph/expression_graph.h) class which builds up an ExpressionGraph of operations, then executes it.

I want to build something with parity in Rust, with more memory-safe conventions. Here I think we can slowly replicate ExpressionGraph in an agentic flow. First I want to build a mechanism to validate the marian-fork as the reference, and then in our agentic system validate our responses with inference-rs against this. Ultimately end to end we run a translations. However in the intermediate flows we validate against things like the internal memory states of the ExpressionGraph and the intermediate nodes. For instance we can 1:1 create Rust implementations of the CPU node operations outside of the primary flow. Then when that's working we can iterate over the ExpressionGraph, for instance getting self attention working first for the internal data representations, and work forward through the graph.
