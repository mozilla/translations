add_executable(translate main.cpp)
target_link_libraries(translate PRIVATE bergamot-translator-source)

# Place the executable in the top-level build directory
set_target_properties(translate PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
