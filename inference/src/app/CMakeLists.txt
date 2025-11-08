add_executable(translator-cli translator_cli.cpp)
target_link_libraries(translator-cli PRIVATE bergamot-translator-source)
