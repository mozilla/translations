#pragma once

/** Main interface for integer matrix multiplication followed by addition of bias for wasm.
 *
 * C = A * B + Bias
 *
 * Input matrix A:
 *   - is a 2-D matrix that typically represents activations as floating point values
 *   - no. of rows should be a multiple of 1 (i.e. no restriction)
 *   - no. of columns should be a multiple of 64
 *   - is represented as array (contiguous memory locations) in row-major format
 *
 * Input matrix B:
 *   - is a 2-D matrix that typically represents fixed model parameters as floating point values
 *   - no. of rows should be:
 *     -- equal to no. of columns of Input matrix A
 *     -- a multiple of 64
 *   - no. of columns should be a multiple of 8
 *   - is represented as array (contiguous memory locations) in row-major format
 *
 *   Please note that it is also possible to pass Input matrix B in 2 more forms:
 *    - One that is already a quantized and transposed version of Input matrix B
 *    - Other that is already a transposed version of Input matrix B
 *
 * Input Bias:
 *   - is an array (contiguous memory locations) that represents bias
 *   - size of the array should be equal to the no. of columns of Input matrix B
 *
 * Output matrix C:
 *   - is a 2-D matrix that represents the result (= A * B + Bias)
 *   - no. of rows will be equal to no. of rows of Input matrix A
 *   - no. of columns will be equal to no. of columns of Input matrix B (in untransposed form)
 *   - is represented as array (contiguous memory locations) in row-major format
 *
 * Please note that most of the functions in this interface might have architecture specific
 * implementations.
 *
 * Conventions followed throughout this file:
 *  - Unless explicitly mentioned, Input matrix B always means an unquantized (i.e. float values)
 *    and non-transposed version
 *  - no. of rows of Input matrix A = `rows_A`
 *  - no. of columns of Input matrix A = no. of rows of Input matrix B = `width`
 *  - no. of columns of Input matrix B = `cols_B`
 */

#include <cstdint>

using Index = uint32_t;

/**
 * Prepare B for the Matrix Multiply function from Input matrix B.
 *
 * Quantization is performed on the input.
 * The final prepared B is in CPU-dependent format and can be used as an input to matrix multiply
 * function (`int8MultiplyAndAddBias`).
 *
 * Please note that this interface might have architecture specific implementation.
 *
 * @param[in]   input_B             An array representing the Input matrix B in row-major format.
 *                                  Size of the array = `width` * `cols_B`.
 *                                  Shape of the matrix: (`width`, `cols_B`)
 * @param[in]   scale               The scaling factor (for quantization)
 * @param[in]   zero_point          The zero point (for quantization)
 * @param[in]   width               No. of rows of Input matrix B. It should be a multiple of 64.
 * @param[in]   cols_B              No. of columns of Input matrix B. It should be a multiple of 8.
 * @param[out]  output              An array representing the prepared B matrix.
 *                                  Size of the array = `width` * `cols_B`.
 */
extern "C" void __attribute__((import_module("wasm_gemm"), import_name("int8_prepare_b")))
int8PrepareB(const float* input_B,
             float scale,
             float zero_point,
             Index width,
             Index cols_B,
             int8_t* output);

/**
 * Prepare B for the Matrix Multiply function from transposed version of Input matrix B.
 *
 * Quantization is performed on floating values of input.
 * The final prepared B is in CPU-dependent format and can be used as an input to matrix multiply
 * function (`int8MultiplyAndAddBias`).
 *
 * Please note that this interface might have architecture specific implementation.
 *
 * @param[in]   input_B_transposed     An array representing transposed version of Input matrix B.
 *                                     It is in column-major format.
 *                                     Size of the array = `width` * `cols_B`.
 *                                     Shape of the matrix: (`cols_B`, `width`)
 * @param[in]   scale                  The scaling factor (for quantization)
 * @param[in]   zero_point             The zero point (for quantization)
 * @param[in]   width                  No. of rows of Input matrix B. It should be a multiple of 64.
 * @param[in]   cols_B                 No. of columns of Input matrix B. Should be a multiple of 8.
 * @param[out]  output                 An array representing the prepared B matrix.
 *                                     Size of the array = `width` * `cols_B`.
 */
extern "C" void
    __attribute__((import_module("wasm_gemm"), import_name("int8_prepare_b_from_transposed")))
    int8PrepareBFromTransposed(const float* input_B_transposed,
                               float scale,
                               float zero_point,
                               Index width,
                               Index cols_B,
                               int8_t* output);

/**
 * Prepare B for the Matrix Multiply function from a quantized and transposed version of Input
 * matrix B which is also in a CPU-independent format.
 *
 * The final prepared B is in CPU-dependent format and can be used as an input to matrix multiply
 * function (`int8MultiplyAndAddBias`).
 *
 * This function is useful while using the quantized models that are stored in a CPU-independent
 * format on the disk.
 *
 * @param[in]   input_B_quant_transposed   An array representing the quantized and transposed
 *                                         version of Input matrix B. It is in column-major format.
 *                                         Size of the array = `width` * `cols_B`.
 *                                         Shape of the matrix: (`cols_B`, `width`)
 * @param[in]   width                      No. of rows of Input matrix B. Should be multiple of 64
 * @param[in]   cols_B                     No. of columns of Input matrix B. Should be multiple of 8
 * @param[out]  output                     An array representing the prepared B matrix.
 *                                         Size of the array = `width` * `cols_B`.
 */
extern "C" void __attribute__((import_module("wasm_gemm"),
                               import_name("int8_prepare_b_from_quantized_transposed")))
int8PrepareBFromQuantizedTransposed(const int8_t* input_B_quant_transposed,
                                    Index width,
                                    Index cols_B,
                                    int8_t* output);

/**
 * Prepare A for the Matrix Multiply function from Input matrix A.
 *
 * It performs quantization on floating values of input.
 * The final prepared A might be architecture dependent. e.g. On some architectures like x86, it
 * might be unsigned (achieved by adding 127 to quantized values) while on others like Arm, it might
 * be signed.
 * The final prepared A can be used as an input to matrix multiply function
 * (`int8MultiplyAndAddBias`).
 *
 * Please note that this interface might have architecture specific implementation.
 *
 * @param[in]   input_A        An array representing the Input matrix A in row-major format.
 *                             Size of the array = `rows_A` * `width`.
 *                             Shape of the matrix: (`rows_A`, `width`)
 * @param[in]   scale          The scaling factor (for quantization)
 * @param[in]   zero_point     The zero point (for quantization)
 * @param[in]   rows_A         No. of rows of Input matrix A. No restriction on its size.
 * @param[in]   width          No. of columns of Input matrix A. It should be a multiple of 64.
 * @param[out]  output         An array representing the prepared A matrix.
 *                             Size of the array = `rows_A` * `width`.
 */
extern "C" void __attribute__((import_module("wasm_gemm"), import_name("int8_prepare_a")))
int8PrepareA(const float* input_A,
             float scale,
             float zero_point,
             Index rows_A,
             Index width,
             int8_t* output);

/**
 * Prepares bias for the Matrix Multiply function.
 *
 * It uses the prepared B (which must be obtained by using any of the int8PrepareB* functions) and
 * a bias input to prepare the final bias.
 *
 * The final bias can be used as an input to matrix multiply function (`int8MultiplyAndAddBias`).
 *
 * @param[in]   input_B_prepared    An array representing the prepared B matrix.
 *                                  Size of the array = `width` * `cols_B`.
 * @param[in]   scale_A             The scaling factor (for quantization) of A
 * @param[in]   zero_point_A        The zero point (for quantization) of A
 * @param[in]   scale_B             The scaling factor (for quantization) of B
 * @param[in]   zero_point_B        The zero point (for quantization) of B
 *                                  factor that is prepared from `scale_A` and `scale_B`.
 * @param[in]   width               No. of rows of Input matrix B (unquantized & non-transposed).
 *                                  It should be a multiple of 64.
 * @param[in]   cols_B              No. of columns of Input matrix B (unquantized & non-transposed)
 *                                  It should be a multiple of 8.
 * @param[in]   input_bias          An array representing the input bias. Size of array = `cols_B`
 * @param[out]  output              An array representing the final prepared bias.
 *                                  Size of the array = `cols_B`
 */
extern "C" void __attribute__((import_module("wasm_gemm"), import_name("int8_prepare_bias")))
int8PrepareBias(const int8_t* input_B_prepared,
                float scale_A,
                float zero_point_A,
                float scale_B,
                float zero_point_B,
                Index width,
                Index cols_B,
                const float* input_bias,
                float* output);

/**
 * Perform multiplication of 2 matrices followed by adding a bias.
 *
 * i.e Output = A_prepared * B_prepared + Bias_prepared
 *
 * The inputs A_prepared, B_prepared and Bias_prepared of this function must be
 * obtained by using `int8PrepareA`, one of the `int8PrepareB*` and `int8PrepareBias`
 * functions respectively.
 *
 * Please note that this interface might have architecture specific implementation.
 *
 * @param[in]   input_A_prepared       An array representing the prepared A matrix.
 *                                     This must be obtained by using `int8PrepareA` function.
 *                                     Size of the array = `rows_A` * `width`.
 * @param[in]   scale_A                The scaling factor (for quantization) of A
 * @param[in]   zero_point_A           The zero point (for quantization) of A
 * @param[in]   input_B_prepared       An array representing the prepared B matrix.
 *                                     This must be obtained by using one of `int8PrepareB*`
 *                                     functions. Size of the array = `width` * `cols_B`.
 * @param[in]   scale_B                The scaling factor (for quantization) of B
 * @param[in]   zero_point_B           The zero point (for quantization) of B
 * @param[in]   input_bias_prepared    An array representing the prepared bias.
 *                                     This must be obtained by using `int8PrepareBias` function.
 *                                     Size of the array = `cols_B`
 * @param[in]   unquant_multiplier     A value that will be multiplied to the final unquantization
 *                                     factor that is prepared from `scale_A` and `scale_B`.
 * @param[in]   rows_A                 No. of rows of Input matrix A. No restriction on its size.
 * @param[in]   width                  No. of columns of Input matrix A (same as no. of columns of
 *                                     Input matrix B). It should be a multiple of 64.
 * @param[in]   cols_B                 No. of columns of Input matrix B. Should be a multiple of 8.
 * @param[out]  output                 An array representing the result matrix in row-major format.
 *                                     Size of the array = `rows_A` * `cols_B`.
 */
extern "C" void
    __attribute__((import_module("wasm_gemm"), import_name("int8_multiply_and_add_bias")))
    int8MultiplyAndAddBias(const int8_t* input_A_prepared,
                           float scale_A,
                           float zero_point_A,
                           const int8_t* input_B_prepared,
                           float scale_B,
                           float zero_point_B,
                           const float* input_bias_prepared,
                           float unquant_multiplier,
                           Index rows_A,
                           Index width,
                           Index cols_B,
                           float* output);

/**
 * Select a subset of columns of prepared B.
 *
 * Indices of the columns to be selected are specified by an array.
 *
 * @param[in]   input_B_prepared   An array representing the prepared B matrix.
 *                                 This must be obtained by using one of the `int8PrepareB*`
 *                                 functions Size of the array = `width` * `cols_B`.
 * @param[in]   width              No. of rows of Input matrix B. It should be a multiple of 64.
 * @param[in]   cols_B             No. of columns of Input matrix B. It should be a multiple of 8.
 * @param[in]   cols               An array of column indices to be selected from prepared B.
 *                                 All indices of the array should be valid. i.e.
 *                                 0 <= cols[N] < cols_B   where N = 0, 1, 2 .... (`num_cols`-1)
 * @param[in]   num_cols           Size of the `cols` array. It should be a multiple of 8.
 * @param[out]  output             An array representing the selected columns of prepared B.
 *                                 Size of the array = `width` * `num_cols`.
 */
extern "C" void __attribute__((import_module("wasm_gemm"), import_name("int8_select_columns_of_b")))
int8SelectColumnsOfB(const int8_t* input_B_prepared,
                     Index width,
                     Index cols_B,
                     const Index* cols,
                     const Index num_cols,
                     int8_t* output);
