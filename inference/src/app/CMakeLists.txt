add_executable(translator-cli translator_cli.cpp)
target_link_libraries(translator-cli PRIVATE bergamot-translator-source)

# Block benchmark: one batched translate per block, model loaded once.
add_executable(block-bench block_bench.cpp)
target_link_libraries(block-bench PRIVATE bergamot-translator-source)
