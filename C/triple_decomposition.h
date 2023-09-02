/**
 * @file
 **/

#ifndef VA_TRIPLE_DECOMPOSITION_H
#define VA_TRIPLE_DECOMPOSITION_H

#include "common_defs.h"
#include "linalg3.h"

/***************************************************************************//**
 * @name Triple decomposition helper functions.
 * @{
 ******************************************************************************/

/***************************************************************************//**
 * @ingroup valib_util
 * Function for determining the Basic Reference Frame (BRF) introduced in
 * \cite Kolar-2007-VIN. This is the coordinate frame in which the shear is
 * maximized and in which it is later eliminated. The BRF is found as an
 * optimization problem, searching the maxima in formula (10) of
 * \cite Kolar-2007-VIN, i.e.
 * \f[
 *    \max_{\alpha \in [0,\pi], \\ \beta \in [0,\pi], \\ \gamma \in [0,\pi/2]}
 *    \left(
 *    |S_{12}\Omega_{12}| + |S_{23}\Omega_{23}| + |S_{31}\Omega_{31}|
 *    \right),
 * \f]
 * where \f$\alpha\f$, \f$\beta\f$, and \f$\gamma\f$ are the three angles of
 * rotation of the coordinate frame.
 * The optimal value is determined by an exhaustive search over all possible
 * values of the angles. The resolution is determined as
 * \f$ h = \pi / n_i \f$,
 * with \f$n_i\f$ the number of intervals (the num_intervals variable).
 * For example, num_intervals = 180 corresponds to 1 degree resolution.
 * This optimization problem can become costly for large number of intervals.
 ******************************************************************************/
VA_DEVICE_FUN void valib_get_brf(VA_DEVICE_ADDR VA_REAL *A, int *num_intervals,
                                 VA_REAL *alpha_brf,
                                 VA_REAL *beta_brf,
                                 VA_REAL *gamma_brf)
{
    int ialpha, ibeta, igamma;

#if defined(CUDA)
    // put Q into shared memory
    extern __shared__ VA_REAL s_data[];
    VA_REAL *Q  = &s_data[0];
#elif defined(OPENCL)
    // put Q into shared memory
    __private VA_REAL Q[9];
#else
    VA_REAL Q[9];  // orthogonal matrices of transformations
#endif
    VA_REAL SO[9]; // symmetric (upper triangle) and antisymmetric (lower
                   // triangle) parts of velocity gradient

    VA_REAL alpha, beta, gamma; // rotation angles
    VA_REAL sina, cosa;         // sin(alpha), cos(alpha)
    VA_REAL sinb, cosb;         // sin(beta),  cos(beta)
    VA_REAL sing, cosg;         // sin(gamma), cos(gamma)

    VA_REAL goal_function;

    // initialize maxima
    VA_REAL goal_brf = 0.;

    // compute stepSize in radians
    VA_REAL stepSize = VA_PI / (VA_REAL) *num_intervals;

    // initialize optimal angles
    *alpha_brf = 0.;
    *beta_brf  = 0.;
    *gamma_brf  = 0.;

    // perform search over all combinations of spherical coordinates
    for (ialpha = 0; ialpha < *num_intervals + 1; ialpha++) {
        alpha = ialpha*stepSize; // alpha in [0,pi]
#if defined(CUDA)
        sincos(alpha, &sina, &cosa);
#elif defined(OPENCL)
        sina = native_sin(alpha);
        cosa = native_cos(alpha);
#else
        sina = sin(alpha);
        cosa = cos(alpha);
#endif
        for (ibeta = 0; ibeta < *num_intervals + 1; ibeta++) {
            beta = ibeta*stepSize; // beta in [0,pi]
#if defined(CUDA)
            sincos(beta, &sinb, &cosb);
#elif defined(OPENCL)
            sinb = native_sin(beta);
            cosb = native_cos(beta);
#else
            sinb = sin(beta);
            cosb = cos(beta);
#endif
            for (igamma = 0; igamma < *num_intervals / 2 + 1; igamma++) {
                gamma = igamma*stepSize; // gamma in [0,pi/2]
#if defined(CUDA)
                sincos(gamma, &sing, &cosg);
#elif defined(OPENCL)
                sing = native_sin(gamma);
                cosg = native_cos(gamma);
#else
                sing = sin(gamma);
                cosg = cos(gamma);
#endif

                // generate matrix Q
#if defined(CUDA)
                __syncthreads();
                if (threadIdx.x == 0) {
#endif
                    valib_constructQ3(sina, cosa, sinb, cosb, sing, cosg, Q);
#if defined(CUDA)
                }
                __syncthreads();
#endif
                // SO = QAQ^T
                valib_tatt3(A, Q, SO);

                // get residual vorticity in 2D
                valib_sym_antisym3(SO);

                // evaluate the goal function
                goal_function = fabs(SO[1]*SO[3])
                              + fabs(SO[2]*SO[6])
                              + fabs(SO[5]*SO[7]);

                // update maxima
                if (goal_function > goal_brf) {
                    goal_brf = goal_function;

                    *alpha_brf = alpha;
                    *beta_brf  = beta;
                    *gamma_brf  = gamma;
                }
            }
        }
    }

    printf("alpha %lf, beta %lf, gamma %lf, goal %lf \n", (*alpha_brf/M_PI*180), (*beta_brf/M_PI*180), (*gamma_brf/M_PI*180), goal_brf);
}

/***************************************************************************//**
 * @}
 ******************************************************************************/

/***************************************************************************//**
 * @ingroup valib_user
 *
 * \brief Triple Decomposition Method (TDM, a.k.a. residual vorticity and
 * residual strain rate)
 * \cite Kolar-2007-VIN, \cite Kolar-2014-RPE
 *
 * Evaluates the norms of the quantities appearing in the triple decomposition
 * of the velocity gradient following \cite Kolar-2007-VIN,
 * \f[
 *    \nabla u = S_{RES} + \Omega_{RES} + (\nabla u)_{SH}
 * \f]
 * where the symmetric tensor \f$ S_{RES} \f$ corresponds to the irrotational
 * straining motion (residual strain rate),
 * \f$ \Omega_{RES} \f$ is the antisymmetric tensor of rigid-body rotation
 * (residual vorticity),
 * and \f$ (\nabla u)_{SH} \f$ contains the effective pure shearing motion.
 * In the Basic Reference Frame (BRF), the last tensor has a strictly asymmetric
 * form, i.e.
 * \f[
 *    u_{i,j} = 0\ \mbox{or}\ u_{j,i} = 0\ \mbox{for all}\ i,j.
 * \f]
 *
 *******************************************************************************
 *
 * \param[in]  A
 *             3x3 matrix of velocity gradient stored column-wise
 *
 * \param[in]  num_intervals
 *             number of steps, i.e. division of the interval \f$ [0,\pi] \f$
 *
 * \param[out] residual_vorticity
 *             Frobenius norm of the tensor of rigid rotation
 *             \f$ \|\Omega_{RES}\|_F \f$
 *
 * \param[out] residual_strain
 *             Frobenius norm of the tensor of irrotational straining
 *             \f$ \|S_{RES}\|_F \f$
 *
 * \param[out] shear
 *             Frobenius norm of the tensor related to the shear contribution
 *             \f$ \|(\nabla u)_{SH}\|_F \f$
 *
 ******************************************************************************/
VA_DEVICE_FUN void valib_triple_decomposition(
    VA_DEVICE_ADDR VA_REAL *A,
    int *num_intervals,
    VA_DEVICE_ADDR VA_REAL *residual_vorticity,
    VA_DEVICE_ADDR VA_REAL *residual_strain,
    VA_DEVICE_ADDR VA_REAL *shear)
{
    VA_REAL SO[9];                       // symmetric (upper triangle) and
                                         // antisymmetric (lower triangle)
                                         // parts of velocity gradient
    VA_REAL Q[9];                        // orthogonal matrices of
                                         // transformations
    VA_REAL alpha_brf, beta_brf, gamma_brf; // angles of basic reference
                                            // frame (BRF)
    VA_REAL sina, cosa;                  // sin(alpha), cos(alpha)
    VA_REAL sinb, cosb;                  // sin(beta),  cos(beta)
    VA_REAL sing, cosg;                  // sin(gamma),  cos(gamma)
    VA_REAL norm_A_2;                    // norm of velocity gradient
    VA_REAL res_tensor_norm_2;           // norm of residual velocity gradient

    int i;
    VA_REAL aux;

    // find BRF angles
    valib_get_brf(A, num_intervals, &alpha_brf, &beta_brf, &gamma_brf);

#if defined(CUDA)
    sincos(alpha_brf, &sina, &cosa);
    sincos(beta_brf,  &sinb, &cosb);
    sincos(gamma_brf, &sing, &cosg);
#elif defined(OPENCL)
    sina = native_sin(alpha_brf);
    cosa = native_cos(alpha_brf);
    sinb = native_sin(beta_brf);
    cosb = native_cos(beta_brf);
    sing = native_sin(gamma_brf);
    cosg = native_cos(gamma_brf);
#else
    sina = sin(alpha_brf);
    cosa = cos(alpha_brf);
    sinb = sin(beta_brf);
    cosb = cos(beta_brf);
    sing = sin(gamma_brf);
    cosg = cos(gamma_brf);
#endif

    // generate matrix Q (this call is not shared for CUDA because this time,
    // each thread has different Q)
    valib_constructQ3(sina, cosa, sinb, cosb, sing, cosg, Q);

    // SO = QAQ^T
    valib_tatt3(A, Q, SO);

    // squared norm of the transformed velocity gradient tensor
    norm_A_2 = 0.;
    for (i = 0; i < 9; i++) {
       norm_A_2 = norm_A_2 + SO[i]*SO[i];
    }

    // Find the residual vorticity inside SO - filter out shear stress
    aux   = valib_min(fabs(SO[3]), fabs(SO[1]));
    SO[3] = valib_sign(aux, SO[3]);
    SO[1] = valib_sign(aux, SO[1]);

    aux   = valib_min(fabs(SO[6]), fabs(SO[2]));
    SO[6] = valib_sign(aux, SO[6]);
    SO[2] = valib_sign(aux, SO[2]);

    aux   = valib_min(fabs(SO[7]), fabs(SO[5]));
    SO[7] = valib_sign(aux, SO[7]);
    SO[5] = valib_sign(aux, SO[5]);

    res_tensor_norm_2 =     SO[0]*SO[0] + SO[4]*SO[4] + SO[8]*SO[8]
                      + 2.*(SO[3]*SO[3] + SO[6]*SO[6] + SO[7]*SO[7]);
    // shear magnitude
    *shear = sqrt(norm_A_2 - res_tensor_norm_2);

    // ( QAQ^T - shear ) is left in SO - find symmetric and antisymmetric parts
    valib_sym_antisym3(SO);

    // ||Omega_RES||_F
    *residual_vorticity = sqrt(  2.*SO[1]*SO[1]
                               + 2.*SO[2]*SO[2]
                               + 2.*SO[5]*SO[5]);

    // ||S_RES||_F
    *residual_strain = sqrt(2.*(SO[3]*SO[3] + SO[6]*SO[6] + SO[7]*SO[7])
                            +  (SO[0]*SO[0] + SO[4]*SO[4] + SO[8]*SO[8]));
}

/***************************************************************************//**
 * @ingroup valib_user
 *
 * \brief Triple Decomposition Method (TDM, a.k.a. residual vorticity and
 * residual strain rate)
 * \cite Kolar-2007-VIN, \cite Kolar-2014-RPE
 *
 * Evaluates the norms of the quantities appearing in the triple decomposition
 * of the velocity gradient following \cite Kolar-2007-VIN,
 * \f[
 *    \nabla u = S_{RES} + \Omega_{RES} + (\nabla u)_{SH}
 * \f]
 * where the symmetric tensor \f$ S_{RES} \f$ corresponds to an irrotational
 * straining motion (residual strain rate),
 * \f$ \Omega_{RES} \f$ is an antisymmetric tensor of rigid-body rotation
 * (residual vorticity), and \f$ (\nabla u)_{SH} \f$ contains the effective pure
 * shearing motion. In the Basic Reference Frame (BRF), the last tensor has
 * a strictly asymmetric form, i.e.
 * \f[
 *    u_{i,j} = 0\ \mbox{or}\ u_{j,i} = 0\ \mbox{for all}\ i,j.
 * \f]
 *
 * The difference from the valib_triple_decomposition function is that
 * the tensor of shear is transformed from BRF to the original coordinate system
 * and there further decomposed into the symmetric and antisymmetric parts,
 * \f[
 *    (\nabla u)_{SH} = S_{SH} + \Omega_{SH}.
 * \f]
 * Norms of these parts are returned by this function individually.
 *
 *******************************************************************************
 *
 * \param[in]  A
 *             3x3 matrix of velocity gradient stored column-wise
 *
 * \param[in]  num_intervals
 *             number of steps, i.e. division of the interval \f$ [0,\pi] \f$
 *
 * \param[out] residual_vorticity
 *             Frobenius norm of the tensor of rigid rotation
 *             \f$ \|\Omega_{RES}\|_F \f$
 *
 * \param[out] residual_strain
 *             Frobenius norm of the tensor of irrotational straining
 *             \f$ \|S_{RES}\|_F \f$
 *
 * \param[out] shear_vorticity
 *             Frobenius norm of the tensor related to the shear contribution
 *             to vorticity
 *             \f$ \|\Omega_{SH}\|_F \f$
 *
 * \param[out] shear_strain
 *             Frobenius norm of the tensor related to the shear contribution
 *             to strain-rate
 *             \f$ \|S_{SH}\|_F \f$
 *
 ******************************************************************************/
VA_DEVICE_FUN void valib_triple_decomposition_4norms(
    VA_DEVICE_ADDR VA_REAL *A, int *num_intervals,
    VA_DEVICE_ADDR VA_REAL *residual_vorticity,
    VA_DEVICE_ADDR VA_REAL *residual_strain,
    VA_DEVICE_ADDR VA_REAL *shear_vorticity,
    VA_DEVICE_ADDR VA_REAL *shear_strain)
{

    VA_REAL SO[9];                       // symmetric (upper triangle) and
                                         // antisymmetric (lower triangle)
                                         // parts of velocity gradient
    VA_REAL QAQ[9];                      // rotated velocity gradient
    VA_REAL Q[9];                        // orthogonal matrices of
                                         // transformations
    VA_REAL alpha_brf, beta_brf, gamma_brf; // angles of basic reference frame
    VA_REAL sina, cosa;                  // sin(alpha), cos(alpha)
    VA_REAL sinb, cosb;                  // sin(beta), cos(beta)
    VA_REAL sing, cosg;                  // sin(gamma), cos(gamma)

    int i;
    VA_REAL aux;

    // find BRF angles
    valib_get_brf(A, num_intervals, &alpha_brf, &beta_brf, &gamma_brf);

#if defined(CUDA)
    sincos(alpha_brf, &sina, &cosa);
    sincos(beta_brf,  &sinb, &cosb);
    sincos(gamma_brf, &sing, &cosg);
#elif defined(OPENCL)
    sina = native_sin(alpha_brf);
    cosa = native_cos(alpha_brf);
    sinb = native_sin(beta_brf);
    cosb = native_cos(beta_brf);
    sing = native_sin(gamma_brf);
    cosg = native_cos(gamma_brf);
#else
    sina = sin(alpha_brf);
    cosa = cos(alpha_brf);
    sinb = sin(beta_brf);
    cosb = cos(beta_brf);
    sing = sin(gamma_brf);
    cosg = cos(gamma_brf);
#endif

    // generate matrix Q ( this call is not shared for CUDA because this time,
    // each thread has different Q )
    valib_constructQ3(sina, cosa, sinb, cosb, sing, cosg, Q);

    // SO = QAQ^T
    valib_tatt3(A, Q, SO);

    // Find the residual vorticity inside SO - filter out shear stress
    aux = valib_min(fabs(SO[3]), fabs(SO[1]));
    SO[3] = valib_sign(aux, SO[3]);
    SO[1] = valib_sign(aux, SO[1]);

    aux = valib_min(fabs(SO[6]), fabs(SO[2]));
    SO[6] = valib_sign(aux, SO[6]);
    SO[2] = valib_sign(aux, SO[2]);

    aux = valib_min(fabs(SO[7]), fabs(SO[5]));
    SO[7] = valib_sign(aux, SO[7]);
    SO[5] = valib_sign(aux, SO[5]);

    // ( QAQ^T - shear ) is left in SO - find symmetric and antisymmetric parts
    valib_sym_antisym3(SO);

    *residual_vorticity = sqrt( 2.*SO[1]*SO[1]
                              + 2.*SO[2]*SO[2]
                              + 2.*SO[5]*SO[5]);

    *residual_strain = sqrt( 2.*(SO[3]*SO[3] + SO[6]*SO[6] + SO[7]*SO[7])
                            +   (SO[0]*SO[0] + SO[4]*SO[4] + SO[8]*SO[8]));

    // SO = QAQ^T
    valib_tatt3(A, Q, QAQ);

    // copy QAQ^T to SO
    valib_mat_copy3(QAQ, SO);

    // Find the residual vorticity inside SO
    aux = valib_min(fabs(SO[3]), fabs(SO[1]));
    SO[3] = valib_sign(aux, SO[3]);
    SO[1] = valib_sign(aux, SO[1]);

    aux = valib_min(fabs(SO[6]), fabs(SO[2]));
    SO[6] = valib_sign(aux, SO[6]);
    SO[2] = valib_sign(aux, SO[2]);

    aux = valib_min(fabs(SO[7]), fabs(SO[5]));
    SO[7] = valib_sign(aux, SO[7]);
    SO[5] = valib_sign(aux, SO[5]);

    // store the difference, i.e. the shear tensor, in SO
    for (i = 0; i < 9; i++) {
       SO[i] = QAQ[i] - SO[i];
    }

    // find the shear strain rate and shear vorticity
    // (QAQ^T - shear) is left in SO - find symmetric and antisymmetric parts
    valib_sym_antisym3(SO);

    // shear_vort = ||Omega_SH||_F
    *shear_vorticity = sqrt(2.*(SO[1]*SO[1] + SO[2]*SO[2] + SO[5]*SO[5]));

    // shear_strain = ||S_SH||_F
    *shear_strain = sqrt(2.*(SO[3]*SO[3] + SO[6]*SO[6] + SO[7]*SO[7])
                         +  (SO[0]*SO[0] + SO[4]*SO[4] + SO[8]*SO[8]));
}

#endif // VA_TRIPLE_DECOMPOSITION_H
