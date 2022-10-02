#ifndef MATRIX_H_
#define MATRIX_H_

void    matrix_multiply(float const* A, int row, int col_a, float const* B, int col_b, float* C) noexcept;

//m should be a square matrix, so, the width should be equal to height
bool    matrix_inv(float const* m, int size, float* inv) noexcept;

void    matrix_transpose(float *m, int row1, int col1, float *mt) noexcept;

#endif