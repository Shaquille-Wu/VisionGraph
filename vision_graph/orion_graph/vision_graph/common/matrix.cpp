#include "matrix.h"
#include <string.h>
#include <math.h>

void matrix_multiply(float const* A, int row, int col_a, float const* B, int col_b, float* C) noexcept
{
    int i = 0, j = 0, k = 0;
    for (i = 0; i < row; ++i) 
    {
        int tetem = i * col_a;
        for (j = 0; j < col_b; ++j) 
        {
            float sum = 0.0f;
            for (k = 0; k < col_a; ++k)
                sum += A[tetem + k] * B[k * col_b + j];
            C[i * col_b + j] = sum;
        }
    }
}

//m should be a square matrix, so, the width should be equal to height
bool    matrix_inv(float const* m, int size, float* inv) noexcept
{
    /*
    * @brief we have three methods to compute the inverse of a matrix
    * a. gauss-jordan elimination
    * b. forward elimination and backward substitution
    * c. determinant and its adjugate matrix
    * here we choose option a
    */
    int   i = 0, j = 0, k = 0;
    float mat_buf[size * size];
    memcpy(mat_buf, m, size * size * sizeof(float));
    memset(inv,     0, size * size * sizeof(float));
    for(i = 0 ; i < size ; i ++)
        inv[i * size + i] = 1.0f;
    
    float max = 0.0f, tmp = 0.0f;
    for (i = 0; i < size; i++)
    {
        max = mat_buf[i * size + i];
        k   = i;
        for (j = i + 1; j < size; j++)
        {
            if (fabsf(mat_buf[j * size + i]) > fabsf(max))
            {
                max = mat_buf[j * size + i];
                k   = j;
            }
        }

        if (k != i)
        {
            for (j = 0; j < size; j++)
            {
                int            idx1 = (i * size) + j;
                int            idx2 = (k * size) + j;
                tmp                 = mat_buf[idx1];
                mat_buf[idx1]       = mat_buf[idx2];
                mat_buf[idx2]       = tmp;

                tmp             = inv[idx1];
                inv[idx1]       = inv[idx2];
                inv[idx2]       = tmp;
            }
        }

        if (fabsf(mat_buf[i * size + i]) <= 1e-6)
            return false;

        tmp = mat_buf[i * size + i];
        for (j = 0; j < size; j++)
        {
            const int  idx = (i * size) + j;
            mat_buf[idx]   = mat_buf[idx] / tmp;
            inv[idx]       = inv[idx] / tmp;
        }
        for (j = 0; j < size; j++)
        {
            if (j != i)
            {
                tmp = mat_buf[(j * size) + i];
                for (k = 0; k < size; k++)
                {
                    const int idx1 = (j * size) + k;
                    const int idx2 = (i * size) + k;
                    mat_buf[idx1]       = mat_buf[idx1] - mat_buf[idx2] * tmp;
                    inv[idx1]           = inv[idx1] - inv[idx2] * tmp;
                }
            }
        }
    }

    return true;
}

void    matrix_transpose(float *m, int row1, int col1, float *mt) noexcept
{
    int i = 0, j = 0;
    for (i = 0; i < col1; ++i)
        for (j = 0; j < row1; ++j)
            mt[i * row1 + j] = m[j * col1 + i];
}