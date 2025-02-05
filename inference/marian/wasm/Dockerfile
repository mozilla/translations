FROM emscripten/emsdk:3.1.8

# Install specific version of CMake
WORKDIR /usr
RUN wget https://github.com/Kitware/CMake/releases/download/v3.17.2/cmake-3.17.2-Linux-x86_64.tar.gz -qO-\
    | tar xzf - --strip-components 1

# Install Python and Java (needed for Closure Compiler minification)
RUN apt-get update \
    && apt-get install -y \
    python3 \
    default-jre

# Necessary for benchmarking
RUN pip3 install sacrebleu
