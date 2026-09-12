/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */


#ifndef ACADOS_SIM_SIM_SDIRK_INTEGRATOR_H_
#define ACADOS_SIM_SIM_SDIRK_INTEGRATOR_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "acados/sim/sim_common.h"
#include "acados/utils/types.h"

#include "blasfeo_common.h"

typedef struct
{
    int nx;
    int nu;
    int nz;
    int np;
    int ny;  // for NLS cost propagation

} sim_sdirk_dims;



typedef struct
{
    /* external functions */
    // implicit fun - can either be fully implicit ode or dae
    //          - i.e. dae has z as additional last argument & nz > 0
    external_function_generic *impl_ode_fun;
    // implicit ode & jac_x & jax_xdot & jac_z
    external_function_generic *impl_ode_fun_jac_x_xdot_z;
    // jax_x & jac_xdot & jac_u & jac_z of implicit ode
    external_function_generic *impl_ode_jac_x_xdot_u_z;
    // hessian of implicit ode:
    external_function_generic *impl_ode_hess;
    // Jacobian of implicit ode w.r.t. p
    external_function_generic *impl_dae_jac_p;

    // for cost propagation
    external_function_generic *nls_y_fun_jac;  // evaluation nls function and jacobian
    external_function_generic *nls_y_fun;  // evaluation nls function
    external_function_generic *conl_cost_fun_jac_hess;
    external_function_generic *conl_cost_fun;

} sdirk_model;



typedef struct
{
    struct blasfeo_dvec *rf;        // residuals of f ((nx+nz))
    struct blasfeo_dvec *k;
    struct blasfeo_dvec *s;
    struct blasfeo_dvec *z;
    struct blasfeo_dvec *xn;        // x at each integration step

    struct blasfeo_dvec *lambda;    // adjoint sensitivities (nx + nu)
    struct blasfeo_dvec *lambdaK;   // auxiliary variable ((nx+nz)*ns) for adjoint propagation

    struct blasfeo_dmat df_dx;     // temporary Jacobian of ode w.r.t x (nx+nz, nx)
    struct blasfeo_dmat df_dxdot;  // temporary Jacobian of ode w.r.t xdot (nx+nz, nx)
    struct blasfeo_dmat df_du;     // temporary Jacobian of ode w.r.t u (nx+nz, nu)
    struct blasfeo_dmat df_dz;     // temporary Jacobian of ode w.r.t z (nx+nz, nu)

    // Jacobian with respect to x and z used in stepwise computation.
    // in case of hessian sensitivity this is ns*nstages mat of size (nx+nz.nx+nz)
    // the first in each stage is actually df w.r.t k and z, due to explicit step
    struct blasfeo_dmat *df_dxz;


    // ipiv: index of pivot vector
    //         if (!opts->sens_hess) - array (ns * (nx + nz)) that is reused
    //         if ( opts->sens_hess) - array (ns * (nx + nz)) * num_steps, to store all
    //              pivot vectors for dG_dxu
    int *ipiv;  // index of pivot vector
} sim_sdirk_workspace;


typedef struct
{
    double *xdot;  // xdot[NX] - initialization for state derivatives k within the integrator
    double *z;     // z[NZ] - initialization for algebraic variables z

    double time_sim;
    double time_ad;
    double time_la;

    double *cost_fun;
    double *outer_hess_is_diag;
    double *cost_scaling_ptr;

    struct blasfeo_dmat *W_chol;  // cholesky factor of weight matrix
    struct blasfeo_dvec *W_chol_diag;
    struct blasfeo_dvec *y_ref;  // y_ref for NLS cost
    struct blasfeo_dvec *cost_grad;
    struct blasfeo_dmat *cost_hess;

    struct blasfeo_dmat *S_p;

} sim_sdirk_memory;


// get & set functions
void sim_sdirk_dims_set(void *config_, void *dims_, const char *field, const int *value);
void sim_sdirk_dims_get(void *config_, void *dims_, const char *field, int* value);

// dims
acados_size_t sim_sdirk_dims_calculate_size();
void *sim_sdirk_dims_assign(void *config_, void *raw_memory);

// model
acados_size_t sim_sdirk_model_calculate_size(void *config, void *dims);
void *sim_sdirk_model_assign(void *config, void *dims, void *raw_memory);
int sim_sdirk_model_set(void *model, const char *field, void *value);

// opts
acados_size_t sim_sdirk_opts_calculate_size(void *config, void *dims);
void *sim_sdirk_opts_assign(void *config, void *dims, void *raw_memory);
void sim_sdirk_opts_initialize_default(void *config, void *dims, void *opts_);
void sim_sdirk_opts_update(void *config_, void *dims, void *opts_);
void sim_sdirk_opts_set(void *config_, void *opts_, const char *field, void *value);

// memory
acados_size_t sim_sdirk_memory_calculate_size(void *config, void *dims, void *opts_);
void *sim_sdirk_memory_assign(void *config, void *dims, void *opts_, void *raw_memory);
int sim_sdirk_memory_set(void *config_, void *dims_, void *mem_, const char *field, void *value);

// workspace
acados_size_t sim_sdirk_workspace_calculate_size(void *config, void *dims, void *opts_);

size_t sim_sdirk_get_external_fun_workspace_requirement(void *config_, void *dims_, void *opts_, void *model_);
void sim_sdirk_set_external_fun_workspaces(void *config_, void *dims_, void *opts_, void *model_, void *workspace_);

void sim_sdirk_config_initialize_default(void *config);

// main
int sim_sdirk(void *config, sim_in *in, sim_out *out, void *opts_, void *mem_, void *work_);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SIM_SIM_ESDIRK_INTEGRATOR_H_
