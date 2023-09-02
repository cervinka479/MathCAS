/**
 * @file
 **/

#ifndef VA_LINALG3_H
#define VA_LINALG3_H

#include "common_defs.h"

/***************************************************************************//**
 * @name Small helper functions
 * @{
 ******************************************************************************/

/***************************************************************************//**
 * @ingroup valib_util
 * Fortran-like SIGN function.
 * Returns absolute value of a with sign of b.
 ******************************************************************************/
VA_DEVICE_FUN VA_REAL valib_sign(VA_REAL a, VA_REAL b)
{
    return (b < (VA_REAL) 0.) ? -fabs(a) : fabs(a);
}

/***************************************************************************//**
 * @ingroup valib_util
 * Function for finding the real cubic root.
 ******************************************************************************/
VA_DEVICE_FUN VA_REAL valib_cubic_root(VA_REAL x)
{
    return valib_sign(pow(fabs(x), 1.0/3.0), x);
}

/***************************************************************************//**
 * @ingroup valib_util
 * Fortran-like MAX function.
 * Returns the larger of the two numbers.
 ******************************************************************************/
VA_DEVICE_FUN VA_REAL valib_max(VA_REAL a, VA_REAL b)
{
    return (a > b) ? a : b;
}

/***************************************************************************//**
 * @ingroup valib_util
 * Fortran-like MIN function.
 * Returns the smaller of the two numbers.
 ******************************************************************************/
VA_DEVICE_FUN VA_REAL valib_min(VA_REAL a, VA_REAL b)
{
    return (a < b) ? a : b;
}

/***************************************************************************//**
 * @ingroup valib_util
 * Simple function for swapping two values.
 ******************************************************************************/
VA_DEVICE_FUN void valib_swap_values(VA_REAL *a, VA_REAL *b)
{
    VA_REAL buf;
    buf = *a;
    *a  = *b;
    *b  = buf;
}

/***************************************************************************//**
 * @ingroup valib_util
 * Function for solving cubic equation using Cardano formulas
 * based on K. Rektorys: Prehled uzite matematiky, p. 39
 * a*x^3 + b*x^2 + c*x + d = 0
 * If determinant is positive, it cannot solve the problem, info = -1.
 * If determinant is negative, it finds the three roots, info = 0.
 ******************************************************************************/
VA_DEVICE_FUN void valib_solve_cubic_equation(VA_REAL a, VA_REAL b,
                                              VA_REAL c, VA_REAL d,
                                              VA_COMPLEX *x1,
                                              VA_COMPLEX *x2,
                                              VA_COMPLEX *x3,
                                              int *info)
{
    int debug = 0;

// define imaginary unit for CUDA
#if defined(CUDA)
    VA_COMPLEX I(0., 1.);
#endif

    VA_REAL q = pow(b/(3.*a),3) - (b*c)/(6.*a*a) + d/(2.*a);
    VA_REAL p = (3*a*c - b*b) / (9*a*a);

    VA_REAL delta = - (p*p*p + q*q);

    if (debug) {
        printf("q:     %lf  \n", q);
        printf("p:     %lf  \n", p);
        printf("delta: %lf  \n", delta);
    }

    if (delta < 0) {
        if (debug) {
           printf("This is a good case with two complex roots.\n");
        }
    }
    else {
        if (debug) {
           printf("Discriminant is positive, this is not a good way to get roots.\n");
        }
        *info = -1;
        return;
    }

    VA_REAL sr = sqrt(-delta);

    if (debug) {
        printf("sr:     %lf  \n", sr);
    }

    VA_REAL mqpsr = -q + sr;
    VA_REAL mqmsr = -q - sr;

    if (debug) {
        printf("mqpsr:     %lf  \n", mqpsr);
        printf("mqmsr:     %lf  \n", mqmsr);
    }

    VA_REAL u = valib_cubic_root(mqpsr);
    VA_REAL v = valib_cubic_root(mqmsr);

    if (debug) {
        printf("fabs(u*v + p):     %lf  \n", fabs(u*v + p));
    }
    if (fabs(u*v + p) > 1.e-5) {
        if (debug) {
            printf("Check of uv = -p failed.");
        }
        *info = -2;
        return;
    }

    VA_COMPLEX e1 = -0.5 + sqrt(3.)/2.*I;
    VA_COMPLEX e2 = -0.5 - sqrt(3.)/2.*I;

    VA_COMPLEX y1 = u + v;
    VA_COMPLEX y2 = e1*u + e2*v;
    VA_COMPLEX y3 = e2*u + e1*v;

    VA_REAL shift = -b/(3.*a);

    *x1 = y1 + shift;
    *x2 = y2 + shift;
    *x3 = y3 + shift;

    // check that ax^3 + bx^2 + cx + d is close to zero for the roots
    VA_COMPLEX check1 = a*(*x1)*(*x1)*(*x1) + b*(*x1)*(*x1) + c*(*x1) + d;
    VA_COMPLEX check2 = a*(*x2)*(*x2)*(*x2) + b*(*x2)*(*x2) + c*(*x2) + d;
    VA_COMPLEX check3 = a*(*x3)*(*x3)*(*x3) + b*(*x3)*(*x3) + c*(*x3) + d;

    if (abs(check1) > 1.e-6) {
        if (debug) {
            printf("Check of x1 failed.");
        }
    }
    if (abs(check2) > 1.e-6) {
        if (debug) {
            printf("Check of x2 failed.");
        }
    }
    if (abs(check3) > 1.e-6) {
        if (debug) {
            printf("Check of x3 failed.");
        }
    }

    // finished correctly
    *info = 0;
}

/***************************************************************************//**
 * @}
 *
 * @name Operations on vectors of length 3
 * @{
 ******************************************************************************/

/***************************************************************************//**
 * @ingroup valib_util
 * Zero a vector of length 3.
 ******************************************************************************/
VA_DEVICE_FUN void valib_vec_zero3(VA_REAL *v)
{
    int i;
    for (i = 0; i < 3; i++) {
        v[i] = 0.;
    }
}

/***************************************************************************//**
 * @ingroup valib_util
 * Function for scalar product of two vectors \f[ result = v_1 \cdot v_2 \f].
 ******************************************************************************/
VA_DEVICE_FUN VA_REAL valib_scalar_prod3(VA_REAL *v1, VA_REAL *v2)
{
    VA_REAL result = (VA_REAL) 0.;
    int i;
    for (i = 0; i < 3; i++)
    {
        result = result + v1[i]*v2[i];
    }
    return result;
}

/***************************************************************************//**
 * @ingroup valib_util
 * Function for cross product of two vectors \f[ v_3 = v_1 \times v_2 \f].
 ******************************************************************************/
VA_DEVICE_FUN void valib_cross_prod3(VA_REAL *v1, VA_REAL *v2, VA_REAL *v3)
{
    v3[0] = v1[1] * v2[2] - v1[2] * v2[1];
    v3[1] = v1[2] * v2[0] - v1[0] * v2[2];
    v3[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

/***************************************************************************//**
 * @}
 *
 * @name Operations on 3x3 matrices
 * @{
 ******************************************************************************/

/***************************************************************************//**
 * @ingroup valib_util
 * Zero a 3x3 matrix.
 ******************************************************************************/
VA_DEVICE_FUN void valib_mat_zero3(VA_REAL *A)
{
    int i, j;
    for (j = 0; j < 3; j++) {
        for (i = 0; i < 3; i++) {
            A[j*3+i] = 0.;
        }
    }
}

/***************************************************************************//**
 * @ingroup valib_util
 * Copy a 3x3 matrix A to a 3x3 matrix B.
 ******************************************************************************/
VA_DEVICE_FUN void valib_mat_copy3(VA_REAL *A, VA_REAL *B)
{
    int i, j;
    for (j = 0; j < 3; j++) {
        for (i = 0; i < 3; i++) {
            B[j*3+i] = A[j*3+i];
        }
    }
}

/***************************************************************************//**
 * @ingroup valib_util
 * Compute Frobenius norm of a 3x3 matrix.
 ******************************************************************************/
VA_DEVICE_FUN void valib_norm_frobenius3(VA_REAL *A, VA_REAL *norm)
{
    *norm = 0.0;
    int i, j;
    for (j = 0; j < 3; j++) {
        for (i = 0; i < 3; i++) {
            *norm = *norm + A[j*3+i]*A[j*3+i];
        }
    }
    *norm = sqrt(*norm);
}

/***************************************************************************//**
 * @ingroup valib_util
 * Compute the trace of a 3x3 matrix (1st tensor invariant).
 ******************************************************************************/
VA_DEVICE_FUN void valib_trace3(VA_REAL *A, VA_REAL *tr3)
{
    *tr3 = A[0] + A[4] + A[8];
}

/***************************************************************************//**
 * @ingroup valib_util
 * Compute the second invariant of a 3x3 matrix (2nd tensor invariant).
 ******************************************************************************/
VA_DEVICE_FUN void valib_second_invariant3(VA_REAL *A,
                                           VA_REAL *second_invariant)
{
    *second_invariant = ( A[0]*A[4] + A[4]*A[8] + A[0]*A[8] )
                      - ( A[1]*A[3] + A[2]*A[6] + A[5]*A[7] );
}

/***************************************************************************//**
 * @ingroup valib_util
 * Compute the determinant of a 3x3 matrix (3rd tensor invariant).
 ******************************************************************************/
VA_DEVICE_FUN void valib_determinant3(VA_REAL *A, VA_REAL *det)
{
    *det = A[0] * A[4] * A[8]
         + A[3] * A[7] * A[2]
         + A[6] * A[1] * A[5]
         - A[0] * A[7] * A[5]
         - A[3] * A[1] * A[8]
         - A[6] * A[4] * A[2];
}

/***************************************************************************//**
 * @ingroup valib_util
 * Transpose A inplace.
 ******************************************************************************/
VA_DEVICE_FUN void valib_transpose3(VA_REAL *A)
{
    valib_swap_values(&A[1], &A[3]);
    valib_swap_values(&A[2], &A[6]);
    valib_swap_values(&A[5], &A[7]);
}

/***************************************************************************//**
 * @ingroup valib_util
 * Find the symmetric and the antisymmetric parts of a 3x3 matrix.
 * Store the symmetric part in the upper triangle, and the antisymmetric part
 * in the lower triangle.
 ******************************************************************************/
VA_DEVICE_FUN void valib_sym_antisym3(VA_REAL *A)
{
    VA_REAL ud, ld;

    int i, j;
    for (i = 0; i < 3; i++) {
        for (j = i+1; j < 3; j++) {
            ud = A[i + j*3];
            ld = A[j + i*3];

            A[i + j*3] = (VA_REAL) 0.5 * (ud + ld);
            A[j + i*3] = (VA_REAL) 0.5 * (ld - ud);
        }
    }
}

/***************************************************************************//**
 * @ingroup valib_util
 * Find the deviatoric part of a 3x3 matrix, i.e. shift the diagonal such that
 * the matrix has zero trace,
 * \f[
 *     A_D = A - \frac{1}{3}\mbox{trace}(A)\ I.
 * \f]
 ******************************************************************************/
VA_DEVICE_FUN void valib_deviatorise3(VA_REAL *A)
{
    // get trace(A)
    VA_REAL trace;
    valib_trace3(A, &trace);

    // subtract a multiple of the third of the trace from the diagonal
    A[0] = A[0] - (VA_REAL) (1./3.) * trace;
    A[4] = A[4] - (VA_REAL) (1./3.) * trace;
    A[8] = A[8] - (VA_REAL) (1./3.) * trace;
}

/***************************************************************************//**
 * @ingroup valib_util
 * Multiplication of a 3x3 matrix with a vector.
 * \f[ y = A x \f].
 ******************************************************************************/
VA_DEVICE_FUN void valib_matvec3_prod(int trans_a, VA_REAL *A,
                                      VA_REAL *x, VA_REAL *y)
{
    int i, j;
    for (i = 0; i < 3; i++) {
        y[i] = 0.0;
        for (j = 0; j < 3; j++) {
            if (trans_a == 0) {
                y[i] = y[i] + A[j*3+i]*x[j];
            }
            else {
                y[i] = y[i] + A[i*3+j]*x[j];
            }
        }
    }
}

/***************************************************************************//**
 * @ingroup valib_util
 * Rank-1 update of a matrix A of dimensions 3x3, computes
 * \f[ A = A + \alpha x y^T \f].
 ******************************************************************************/
VA_DEVICE_FUN void valib_rank1_update3(VA_REAL alpha,
                                       VA_REAL *x, VA_REAL *y,
                                       VA_REAL *A)
{
    int ind;
    VA_REAL tmp;

    ind = 0;
    int i, j;
    for (j = 0; j < 3; j++) {
        tmp = alpha * y[j];
        for (i = 0; i < 3; i++) {
            A[ind] = A[ind] + x[i]*tmp;
            ind = ind + 1;
        }
    }
}

/***************************************************************************//**
 * @ingroup valib_util
 * Transformation of a square 3x3 matrix A
 * \f[ TATT = T A T^T \f].
 ******************************************************************************/
VA_DEVICE_FUN void valib_tatt3(VA_REAL *A, VA_REAL *T, VA_REAL *TATT)
{
    unsigned i, j, ind;

    // zero TATT
    valib_mat_zero3(TATT);

    // perform series of rank one updates
    ind = 0;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            // call rank one update on each element of A
            valib_rank1_update3(A[ind], &T[j*3], &T[i*3], TATT);
            ind = ind + 1;
        }
    }
}

/***************************************************************************//**
 * @ingroup valib_util
 * Multiplication of two 3x3 matrices.
 * \f[ C = A B + C \f].
 ******************************************************************************/
VA_DEVICE_FUN void valib_matmat3_prod(int trans_a, int trans_b,
                                      VA_REAL *A, VA_REAL *B, VA_REAL *C)
{
    int i, j, k;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 3; k++) {
                if      (trans_a == 0 && trans_b == 0) {
                    C[j*3+i] = C[j*3+i] + A[k*3+i] * B[j*3+k];
                }
                else if (trans_a != 0 && trans_b == 0) {
                    C[j*3+i] = C[j*3+i] + A[i*3+k] * B[j*3+k];
                }
                else if (trans_a == 0 && trans_b != 0) {
                    C[j*3+i] = C[j*3+i] + A[k*3+i] * B[k*3+j];
                }
                else {//(trans_a != 0 && trans_b != 0)
                    C[j*3+i] = C[j*3+i] + A[i*3+k] * B[k*3+j];
                }
            }
        }
    }
}

/***************************************************************************//**
 * @ingroup valib_util
 * Gram-Schmidt orthogonalization of a 3x3 matrix.
 ******************************************************************************/
VA_DEVICE_FUN void valib_gram_schmidt3(VA_REAL *Q)
{
    VA_REAL factor;

    // column-wise process
    int i, j, k;
    for (j = 0; j < 3; j++) {
        for (i = 0; i < j; i++) {
            // compute the scalar product with previous columns
            factor = valib_scalar_prod3(&Q[j*3], &Q[i*3]);

            // subtract the projected vector
            for (k = 0; k < 3; k++) {
                Q[j*3+k] = Q[j*3+k] - factor * Q[i*3+k];
            }
        }

        // normalize the column
        factor = valib_scalar_prod3(&Q[j*3], &Q[j*3]);
        for (k = 0; k < 3; k++) {
            Q[j*3+k] = Q[j*3+k] / sqrt(factor);
        }
    }
}

/***************************************************************************//**
 * @ingroup valib_util
 * Computing eigenvalues. Uses the method given in \cite Smith-1961-EST.
 ******************************************************************************/
VA_DEVICE_FUN void valib_eigenvalues_sym3(VA_REAL *A, VA_REAL *eigval)
{
    // compute first invariant
    VA_REAL Idev;
    valib_trace3(A, &Idev);
    VA_REAL IdevMod = Idev / 3.;

    // compute deviator
    VA_REAL dev[9];
    valib_mat_copy3(A, dev);
    dev[0] = dev[0] - IdevMod;
    dev[4] = dev[4] - IdevMod;
    dev[8] = dev[8] - IdevMod;

    // compute second invariant of deviator
    VA_REAL IIdev;
    valib_second_invariant3(dev, &IIdev);
    VA_REAL IIdevMod = -IIdev / 3.;

    // compute half the determinant (third invariant) of deviator
    VA_REAL IIIdev;
    valib_determinant3(dev, &IIIdev);
    VA_REAL IIIdevMod = 0.5 * IIIdev;

    // compute strange angle
    VA_REAL aux =  IIdevMod*IIdevMod*IIdevMod - IIIdevMod*IIIdevMod;
    VA_REAL phi;
    if (aux < 0. || fabs(IIIdevMod) < 1.e-15) {
        phi = 0.;
    }
    else {
        phi = atan( sqrt(aux) / IIIdevMod ) / 3.;
    }

    // readjust
    if (phi < 0.) phi += VA_PI / 3.;

    // eigenvalues
    eigval[0] = IdevMod + 2.*sqrt(IIdevMod) * cos(phi);
    eigval[1] = IdevMod - sqrt(IIdevMod) * (cos(phi) + sqrt(3.) * sin(phi));
    eigval[2] = IdevMod - sqrt(IIdevMod) * (cos(phi) - sqrt(3.) * sin(phi));

    // sort eigenvalues
    if (eigval[0] > eigval[1] ) valib_swap_values( &eigval[0],  &eigval[1] );
    if (eigval[1] > eigval[2] ) valib_swap_values( &eigval[1],  &eigval[2] );
    if (eigval[0] > eigval[1] ) valib_swap_values( &eigval[0],  &eigval[1] );

    return;
}

/***************************************************************************//**
 * @ingroup valib_util
 * Find a the index of the pivot in the column of a matrix. It searches the
 * jcol-th column of matrix A from index start to 3, i.e. index of the maximum
 * of absolute value of A(start:,jcol).
 ******************************************************************************/
VA_DEVICE_FUN int valib_find_pivot3(VA_REAL *A, int jcol, int start)
{
    int debug = 0;
    int ipiv = start;
    for (int i = start+1; i<3; i++) {
        if (fabs(A[i+jcol*3]) > fabs(A[ipiv+jcol*3]))
            ipiv = i;
    }

    VA_REAL numerical_zero = 1.0e-6;
    if (fabs(A[ipiv+jcol*3]) < numerical_zero) {
        if (debug)
            printf("Warning: no nonzero pivot in column %d.\n", jcol);

        ipiv = -1;
    }

    return ipiv;
}

/***************************************************************************//**
 * @ingroup valib_util
 * Find a nontrivial solution to the equation Ax = 0. This function is used
 * for searching the eigenvector corresponding to real eigenvalue in case of
 * simple eigenvalues. It expects one-dimensional nullspace, otherwise returns
 * an error.
 ******************************************************************************/
VA_DEVICE_FUN void valib_find_nullspace3(VA_REAL *A, VA_REAL *x, int *info)
{
    int debug = 0;

    // zero the vector
    x[0] = 0.; x[1] = 0.; x[2] = 0.;

    // handle special cases with zero column
    int num_zero_cols = 0;
    int i_zero_col = -1;
    int ip;
    for (int ji = 0; ji < 3; ji++) {
        int j = 2-ji;
        ip = valib_find_pivot3(A,j,0);
        if (ip == -1) {
            if (debug) {
               printf("There is no meaningful pivot in the %d-th column.\n",j);
            }
            num_zero_cols++;
            i_zero_col = j;
        }
    }

    if (num_zero_cols > 0) {
        if (num_zero_cols > 1) {
            if (debug) {
               printf("Oops! The matrix has more than one zero columns. I search only for 1-dimensional nullspace.\n");
            }
            *info = -1;
            return;
        }
        else {
            x[i_zero_col] = 1.;
            *info = 0;
            return;
        }
    }

    // in the usual case, permute the rows and eliminate first column
    if (ip != 0) {
        for (int j = 0; j < 3; j++) {
            valib_swap_values(&A[j*3], &A[ip+j*3]);
        }
    }

    if (debug) {
        printf("Matrix in GEM after 1st swap: \n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                printf(" %lf ", A[i+j*3]);
            }
            printf("\n");
        }
    }

    VA_REAL coef;

    // eliminate the first column using the pivot
    for (int i = 1; i < 3; i++) {
        coef = A[i] / A[0];
        for (int j = 0; j < 3; j++) {
            A[i+j*3] -= coef*A[j*3];
        }
    }

    if (debug) {
        printf("Matrix in GEM without 1st col: \n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                printf(" %lf ", A[i+j*3]);
            }
            printf("\n");
        }
    }

    // now work on the second column
    int jpiv, jfree;
    ip = valib_find_pivot3(A,1,1);
    if (ip == -1) {
        // the second column is not a pivot one, we can search for the nullspace
        jpiv  = 2;
        jfree = 1;
    }
    else {
        jpiv  = 1;
        jfree = 2;
    }

    ip = valib_find_pivot3(A,jpiv,1);
    if (ip == -1) {
        if (debug) {
           printf("Something is wrong with the matrix, the defect seems larger than one.");
        }
        *info = -2;
        return;
    }
    if (ip == 2) {
        // swap the rows
        for (int j = 0; j < 3; j++) {
            valib_swap_values(&A[1+j*3], &A[2+j*3]);
        }
    }

    // eliminate the last row
    coef = A[2+jpiv*3] / A[1+jpiv*3];
    for (int j = 0; j < 3; j++) {
        A[2+j*3] -= coef*A[1+j*3];
    }
    // eliminate the first row
    coef = A[0+jpiv*3] / A[1+jpiv*3];
    for (int j = 0; j < 3; j++) {
        A[j*3] -= coef*A[1+j*3];
    }

    // check that the last row, in particular the last entry, is zero
    if (valib_find_pivot3(A,2,2) != -1) {
        if (debug) {
            printf("Something is wrong, the matrix seems to be full rank.\n");
        }
        *info = -2;
        return;
    }

    if (debug) {
        printf("Matrix in GEM before dividing by diagonal: \n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                printf(" %lf ", A[i+j*3]);
            }
            printf("\n");
        }
    }

    // finalize the Gauss-Jordan elimination by dividing with the diagonal values
    coef = A[0];
    for (int j = 0; j < 3; j++) {
        A[j*3]   /= coef;
    }
    coef = A[1+jpiv*3];
    for (int j = 0; j < 3; j++) {
        A[1+j*3] /= coef;
    }

    if (debug) {
        printf("Matrix in GEM in the upper right echelon form:\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                printf(" %d %lf ", i+j*3, A[i+j*3]);
            }
            printf("\n");
        }
    }

    // fill the nullspace basis vector
    x[jfree] = 1.;
    x[jpiv]  = -A[jpiv+jfree*3];
    x[0]     = -A[jfree*3];

    // normalize the basis vector
    VA_REAL norm = sqrt(valib_scalar_prod3(&x[0], &x[0]));
    for (int i = 0; i < 3; i++) {
        x[i] /= norm;
    }

    *info = 0;
}


/***************************************************************************//**
 * @ingroup valib_util
 * Get orthogonal matrix Q column-wise based on sines and cosines of 3 angles
 * of rotations of coordinate system along coordinate axes by z, y', and z'',
 * see \cite Kolar-2007-VIN.
 ******************************************************************************/
VA_DEVICE_FUN void valib_constructQ3(VA_REAL sina, VA_REAL cosa,
                                     VA_REAL sinb, VA_REAL cosb,
                                     VA_REAL sing, VA_REAL cosg,
                                     VA_REAL *Q )
{
    Q[0] = cosa*cosb*cosg - sina*sing;
    Q[1] = -cosa*cosb*sing - sina*cosg;
    Q[2] = cosa*sinb;
    Q[3] = sina*cosb*cosg + cosa*sing;
    Q[4] = -sina*cosb*sing + cosa*cosg;
    Q[5] = sina*sinb;
    Q[6] = -sinb*cosg;
    Q[7] = sinb*sing;
    Q[8] = cosb;
}

/***************************************************************************//**
 * @ingroup valib_util
 * Get orthogonal matrix Q column-wise based on components of a normal vector n
 * which corresponds to the last row of Q.
 * Vector n need not be normalized to \f$ \|n\|_2 = 1\f$.
 ******************************************************************************/
VA_DEVICE_FUN void valib_constructQfromN3(VA_REAL *n, VA_REAL *Q)
{
    VA_REAL vec[3];
    VA_REAL factor;

    // initialize the first row of Q with n
    Q[0] = n[0]; Q[1] = n[1]; Q[2] = n[2];

    // fill some entries to initialize the other entries in matrix Q
    Q[3] = n[1]; Q[4] = n[2]; Q[5] = n[0];
    Q[6] = n[2]; Q[7] = n[0]; Q[8] = n[1];

    // perform Gram-Schmidt orthogonalization
    valib_gram_schmidt3(Q);

    // swap the first and the third column - at the moment, the matrix contains
    // | z y x |
    int i;
    for (i = 0; i < 3; i++) {
        valib_swap_values(&Q[i], &Q[6+i]);
    }

    // verify that the matrix has a right-hand oriented coordinate system,
    // Q_1 x Q_2 = Q_3
    valib_cross_prod3(&Q[0], &Q[3], vec);

    // revert sign of the new x-axis to keep right-hand oriented coordinate
    // system
    factor = valib_scalar_prod3(vec, &Q[6]);
    if (factor < 0.) {
        for (i = 0; i < 3; i++) {
            Q[i] = -Q[i];
        }
    }

    // transpose the matrix
    valib_transpose3(Q);
}

/***************************************************************************//**
 * @}
 ******************************************************************************/

#endif // VA_LINALG3_H
