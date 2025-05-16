# inference/marian is tested elsewhere in both paths, turning off here.
set(COMPILE_TESTS OFF)

if(COMPILE_WASM)
  # This is a bad way of adding compilation flags. Will be improved soon.
  add_compile_options(${WASM_COMPILE_FLAGS})
  add_link_options(${WASM_LINK_FLAGS})
endif(COMPILE_WASM)


# Add include directories for 3rd party targets to be able to use it anywhere in the
# project without explicitly specifying their include directories. Once they
# fixe this problem, it can be removed.

if(NOT COMPILE_WASM)
  add_subdirectory(ssplit-cpp EXCLUDE_FROM_ALL)
  get_property(INCLUDE_DIRECTORIES DIRECTORY ssplit-cpp/src PROPERTY INCLUDE_DIRECTORIES)
  target_include_directories(ssplit PUBLIC ${INCLUDE_DIRECTORIES})
endif(NOT COMPILE_WASM)
