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
get_property(INCDIRS DIRECTORY browsermt-marian-dev/src PROPERTY INCLUDE_DIRECTORIES)
target_include_directories(marian PUBLIC ${INCDIRS})

if(NOT COMPILE_WASM)
  add_subdirectory(ssplit-cpp EXCLUDE_FROM_ALL)
  get_property(INCLUDE_DIRECTORIES DIRECTORY ssplit-cpp/src PROPERTY INCLUDE_DIRECTORIES)
  target_include_directories(ssplit PUBLIC ${INCLUDE_DIRECTORIES})
endif(NOT COMPILE_WASM)

get_property(COMPILE_DEFINITIONS DIRECTORY browsermt-marian-dev PROPERTY COMPILE_DEFINITIONS) 
target_compile_definitions(marian PUBLIC ${COMPILE_DEFINITIONS})

get_property(COMPILE_OPTIONS DIRECTORY browsermt-marian-dev PROPERTY COMPILE_OPTIONS) 
target_compile_options(marian PUBLIC ${COMPILE_OPTIONS})

# Compilation flags 
get_directory_property(CMAKE_C_FLAGS DIRECTORY browsermt-marian-dev DEFINITION CMAKE_C_FLAGS) 
get_directory_property(CMAKE_CXX_FLAGS DIRECTORY browsermt-marian-dev DEFINITION CMAKE_CXX_FLAGS) 
set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} PARENT_SCOPE)    
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} PARENT_SCOPE)
