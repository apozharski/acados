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


#include "acados/sim/sim_esdirk_integrator.h"

// standard
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// acados
#include "acados/utils/mem.h"
#include "acados/utils/print.h"
#include "acados/utils/math.h"

#include "acados/sim/sim_common.h"

#include "blasfeo_d_aux.h"
#include "blasfeo_d_blas.h"
#include "blasfeo_common.h"


/************************************************
 * dims
 ************************************************/

acados_size_t sim_esdirk_dims_calculate_size()
{
    acados_size_t size = sizeof(sim_esdirk_dims);

    return size;
}

void *sim_esdirk_dims_assign(void *config_, void *raw_memory)
{
    char *c_ptr = raw_memory;

    sim_esdirk_dims *dims = (sim_esdirk_dims *) c_ptr;
    c_ptr += sizeof(sim_esdirk_dims);

    dims->nx = 0;
    dims->nu = 0;
    dims->nz = 0;
    dims->ny = 0;
    dims->np = 0;

    assert((char *) raw_memory + sim_esdirk_dims_calculate_size() >= c_ptr);

    return dims;
}



void sim_esdirk_dims_set(void *config_, void *dims_, const char *field, const int *value)
{
    sim_esdirk_dims *dims = dims_;

    if (!strcmp(field, "nx"))
    {
        dims->nx = *value;
    }
    else if (!strcmp(field, "nu"))
    {
        dims->nu = *value;
    }
    else if (!strcmp(field, "nz"))
    {
        dims->nz = *value;
    }
    else if (!strcmp(field, "ny"))
    {
        dims->ny = *value;
    }
    else if (!strcmp(field, "np"))
    {
        dims->np = *value;
    }
    else if (!strcmp(field, "np_global"))
    {
        // np_global dimension not needed
    }
    else
    {
        printf("\nerror: sim_esdirk_dims_set: field not available: %s\n", field);
        exit(1);
    }
}



void sim_esdirk_dims_get(void *config_, void *dims_, const char *field, int *value)
{
    sim_esdirk_dims *dims = dims_;

    if (!strcmp(field, "nx"))
    {
        *value = dims->nx;
    }
    else if (!strcmp(field, "nu"))
    {
        *value = dims->nu;
    }
    else if (!strcmp(field, "nz"))
    {
        *value = dims->nz;
    }
    else if (!strcmp(field, "np"))
    {
        *value = dims->np;
    }
    else
    {
        printf("\nerror: sim_esdirk_dims_get: field not available: %s\n", field);
        exit(1);
    }
}



/************************************************
 * model
 ************************************************/

acados_size_t sim_esdirk_model_calculate_size(void *config, void *dims)
{
    acados_size_t size = 0;

    size += sizeof(esdirk_model);

    return size;
}



void *sim_esdirk_model_assign(void *config, void *dims, void *raw_memory)
{
    char *c_ptr = (char *) raw_memory;

    esdirk_model *model = (esdirk_model *) c_ptr;
    c_ptr += sizeof(esdirk_model);

    model->impl_ode_fun = NULL;
    model->impl_ode_fun_jac_x_xdot_z = NULL;
    model->impl_ode_jac_x_xdot_u_z = NULL;
    model->impl_dae_jac_p = NULL;
    model->impl_ode_hess = NULL;

    assert((char *) raw_memory + sim_esdirk_model_calculate_size(config, dims) >= c_ptr);

    return model;
}



int sim_esdirk_model_set(void *model_, const char *field, void *value)
{
    esdirk_model *model = model_;

    if (!strcmp(field, "impl_ode_fun") || !strcmp(field, "impl_dae_fun"))
    {
        model->impl_ode_fun = value;
    }
    else if (!strcmp(field, "impl_ode_fun_jac_x_xdot") || !strcmp(field, "impl_dae_fun_jac_x_xdot"))
    {
        // TODO(oj): remove this case and fix dependencies
        model->impl_ode_fun_jac_x_xdot_z = value;
    }
    else if (!strcmp(field, "impl_ode_fun_jac_x_xdot_z") || !strcmp(field, "impl_dae_fun_jac_x_xdot_z"))
    {
        model->impl_ode_fun_jac_x_xdot_z = value;
    }
    else if (!strcmp(field, "impl_ode_jac_x_xdot_u") || !strcmp(field, "impl_dae_jac_x_xdot_u"))
    {
        // TODO(oj): remove this and update with z everywhere
        model->impl_ode_jac_x_xdot_u_z = value;
    }
    else if (!strcmp(field, "impl_ode_jac_x_xdot_u_z") || !strcmp(field, "impl_dae_jac_x_xdot_u_z"))
    {
        model->impl_ode_jac_x_xdot_u_z = value;
    }
    else if (!strcmp(field, "impl_dae_jac_p"))
    {
        model->impl_dae_jac_p = value;
    }
    else if (!strcmp(field, "impl_ode_hes") || !strcmp(field, "impl_ode_hess") || !strcmp(field, "impl_dae_hess"))
    {
        model->impl_ode_hess = value;
    }
    else if (!strcmp(field, "nls_y_fun_jac") )
    {
        model->nls_y_fun_jac = value;
    }
    else if (!strcmp(field, "nls_y_fun") )
    {
        model->nls_y_fun = value;
    }
    else if (!strcmp(field, "conl_cost_fun_jac_hess") )
    {
        model->conl_cost_fun_jac_hess = value;
    }
    else if (!strcmp(field, "conl_cost_fun") )
    {
        model->conl_cost_fun = value;
    }
    else
    {
        printf("\nerror: sim_esdirk_model_set: wrong field: %s\n", field);
        exit(1);
    }

    return ACADOS_SUCCESS;
}



/************************************************
 * opts
 ************************************************/

acados_size_t sim_esdirk_opts_calculate_size(void *config_, void *dims)
{
    int ns_max = NS_MAX;

    acados_size_t size = 0;

    size += sizeof(sim_opts);

    size += ns_max * ns_max * sizeof(double);  // A_mat
    size += ns_max * sizeof(double);           // b_vec
    size += ns_max * sizeof(double);           // c_vec

    size += butcher_tableau_work_calculate_size(ns_max);

    make_int_multiple_of(8, &size);
    size += 1 * 8;

    return size;
}

void *sim_esdirk_opts_assign(void *config_, void *dims, void *raw_memory)
{
    int ns_max = NS_MAX;

    char *c_ptr = (char *) raw_memory;

    sim_opts *opts = (sim_opts *) c_ptr;
    c_ptr += sizeof(sim_opts);

    align_char_to(8, &c_ptr);

    // work
    opts->work = c_ptr;
    c_ptr += butcher_tableau_work_calculate_size(ns_max);

    assign_and_advance_double(ns_max * ns_max, &opts->A_mat, &c_ptr);
    assign_and_advance_double(ns_max, &opts->b_vec, &c_ptr);
    assign_and_advance_double(ns_max, &opts->c_vec, &c_ptr);

    assert((char *) raw_memory + sim_esdirk_opts_calculate_size(config_, dims) >= c_ptr);

    return (void *) opts;
}



void sim_esdirk_opts_initialize_default(void *config_, void *dims_, void *opts_)
{
    sim_esdirk_dims *dims = (sim_esdirk_dims *) dims_;
    sim_opts *opts = opts_;

    // default options
    opts->newton_iter = 3;
    // opts->scheme = NULL;
    opts->num_steps = 2;
    opts->num_forw_sens = dims->nx + dims->nu;
    opts->sens_forw = false; // TODO(@anton)
    opts->sens_adj = false;
    opts->sens_hess = false;
    opts->jac_reuse = false; // TODO(@anton)
    opts->exact_z_output = false;
    opts->ns = 4;
    opts->collocation_type = ESDIRK;
    opts->newton_tol = 0.0;

    assert(opts->ns <= NS_MAX && "ns > NS_MAX!");

    // butcher tableau
    calculate_butcher_tableau(opts->ns, opts->collocation_type, opts->c_vec, opts->b_vec, opts->A_mat, opts->work);
    // for consistency check
    opts->tableau_size = opts->ns;
    opts->cost_computation = false;

    // TODO(oj): check if constr h or cost depend on z, turn on in this case only.
    if (dims->nz > 0)
    {
        opts->output_z = true;
        opts->sens_algebraic = false; // TODO(@anton)
    }
    else
    {
        opts->output_z = false;
        opts->sens_algebraic = false;
    }

    return;
}



void sim_esdirk_opts_update(void *config_, void *dims, void *opts_)
{
    sim_opts *opts = opts_;

    assert(opts->ns <= NS_MAX && "ns > NS_MAX!");

    calculate_butcher_tableau(opts->ns, opts->collocation_type, opts->c_vec, opts->b_vec, opts->A_mat, opts->work);

    opts->tableau_size = opts->ns;

    // for debugging: print butcher tableau
    // printf("Butcher tableau\n");
    // printf("\nc_vec:\n");
    // for (int i = 0; i < opts->ns; i++)
    // {
    //     printf("%f\t", opts->c_vec[i]);
    // }
    // printf("\nb_vec:\n");
    // for (int i = 0; i < opts->ns; i++)
    // {
    //     printf("%f\t", opts->b_vec[i]);
    // }
    // printf("\nA_mat:\n");
    // d_print_mat(opts->ns, opts->ns, opts->A_mat, opts->ns);

    return;
}



void sim_esdirk_opts_set(void *config_, void *opts_, const char *field, void *value)
{
    sim_opts *opts = (sim_opts *) opts_;
    sim_opts_set_(opts, field, value);
}



void sim_esdirk_opts_get(void *config_, void *opts_, const char *field, void *value)
{
    sim_opts *opts = (sim_opts *) opts_;
    sim_opts_get_(config_, opts, field, value);
}



/************************************************
 * memory
 ************************************************/

acados_size_t sim_esdirk_memory_calculate_size(void *config, void *dims_, void *opts_)
{
    // typecast
    sim_esdirk_dims *dims = (sim_esdirk_dims *) dims_;
    sim_opts *opts = opts_;

    // necessary integers
    int nx = dims->nx;
    int nz = dims->nz;
    int nu = dims->nu;

    acados_size_t size = sizeof(sim_esdirk_memory);

    size += nx * sizeof(double); // xdot
    size += nz * sizeof(double); // z
    size += 8;  // corresponds to memory alignment TODO(@anton) what????? is this manual mem allignment i.e. is this not... machine dependent?

    if (opts->cost_computation)
    {
        size += 1 * sizeof(struct blasfeo_dmat);  // cost_hess
        size += 1 * blasfeo_memsize_dmat(nx+nu, nx+nu);  // cost_hess
        size += 64;
    }

    if (opts->sens_forw_p)
    {
        size += 1 * blasfeo_memsize_dmat(nx, dims->np);
        size += 1 * sizeof(struct blasfeo_dmat);
        size += 64;
    }

    make_int_multiple_of(8, &size);

    return size;
}


// TODO(@anton) update this as needed, I currently don't support most options ;)
void *sim_esdirk_memory_assign(void *config, void *dims_, void *opts_, void *raw_memory)
{
    char *c_ptr = (char *) raw_memory;

    // typecast
    sim_esdirk_dims *dims = (sim_esdirk_dims *) dims_;
    sim_opts *opts = opts_;

    // necessary integers
    int nx = dims->nx;
    int nu = dims->nu;
    int nz = dims->nz;

    // struct
    sim_esdirk_memory *mem = (sim_esdirk_memory *) c_ptr;
    c_ptr += sizeof(sim_esdirk_memory);

    align_char_to(8, &c_ptr);
    if (opts->cost_computation)
    {
        assign_and_advance_blasfeo_dmat_structs(1, &mem->cost_hess, &c_ptr);
    }

    mem->S_p = NULL;

    if (opts->sens_forw_p)
    {
        assign_and_advance_blasfeo_dmat_structs(1, &mem->S_p, &c_ptr);
        align_char_to(64, &c_ptr);
        assign_and_advance_blasfeo_dmat_mem(nx, dims->np, mem->S_p, &c_ptr);
        blasfeo_dgese(nx, dims->np, 0.0, mem->S_p, 0, 0);
    }

    // assign doubles
    assign_and_advance_double(nz, &mem->z, &c_ptr);
    assign_and_advance_double(nx, &mem->xdot, &c_ptr);

    if (opts->cost_computation)
    {
        align_char_to(64, &c_ptr);
        assign_and_advance_blasfeo_dmat_mem(nx+nu, nx+nu, mem->cost_hess, &c_ptr);
    }

    // initialization of xdot, z is 0 if not changed
    for (int ii = 0; ii < nx; ii++)
        mem->xdot[ii] = 0.0;
    for (int ii = 0; ii < nz; ii++)
        mem->z[ii] = 0.0;

    return mem;
}



int sim_esdirk_memory_set(void *config_, void *dims_, void *mem_, const char *field, void *value)
{
    sim_config *config = config_;
    sim_esdirk_memory *mem = (sim_esdirk_memory *) mem_;

    int status = ACADOS_SUCCESS;

    if (!strcmp(field, "xdot"))
    {
        int nx;
        config->dims_get(config_, dims_, "nx", &nx);
        double *xdot = value;
        for (int ii=0; ii < nx; ii++)
            mem->xdot[ii] = xdot[ii];
    }
    else if (!strcmp(field, "z"))
    {
        int nz;
        config->dims_get(config_, dims_, "nz", &nz);
        double *z = value;
        for (int ii=0; ii < nz; ii++)
            mem->z[ii] = z[ii];
    }
    else if (!strcmp(field, "cost_fun"))
    {
        mem->cost_fun = value;
    }
    else if (!strcmp(field, "cost_grad"))
    {
        mem->cost_grad = value;
    }
    else if (!strcmp(field, "W_chol"))
    {
        mem->W_chol = value;
    }
    else if (!strcmp(field, "W_chol_diag"))
    {
        mem->W_chol_diag = value;
    }
    else if (!strcmp(field, "outer_hess_is_diag"))
    {
        mem->outer_hess_is_diag = value;
    }
    else if (!strcmp(field, "y_ref"))
    {
        mem->y_ref = value;
    }
    else if (!strcmp(field, "cost_scaling_ptr"))
    {
        mem->cost_scaling_ptr = value;
    }
    else if (!strcmp(field, "guesses_blasfeo"))
    {
        int nx, nz;
        config->dims_get(config_, dims_, "nx", &nx);
        config->dims_get(config_, dims_, "nz", &nz);

        struct blasfeo_dvec *sim_guess = (struct blasfeo_dvec *) value;
        blasfeo_unpack_dvec(nx, sim_guess, 0, mem->xdot, 1);
        blasfeo_unpack_dvec(nz, sim_guess, nx, mem->z, 1);
    }
    else
    {
        printf("sim_esdirk_memory_set: field %s is not supported! \n", field);
        exit(1);
    }

    return status;
}



int sim_esdirk_memory_set_to_zero(void *config_, void * dims_, void *opts_, void *mem_)
{
    sim_config *config = config_;
    sim_esdirk_memory *mem = (sim_esdirk_memory *) mem_;

    int status = ACADOS_SUCCESS;

    int nx, nz;
    config->dims_get(config_, dims_, "nz", &nz);
    config->dims_get(config_, dims_, "nx", &nx);

    for (int ii=0; ii < nz; ii++)
        mem->z[ii] = 0.0;
    for (int ii=0; ii < nx; ii++)
        mem->xdot[ii] = 0.0;

    return status;
}



void sim_esdirk_memory_get(void *config_, void *dims_, void *mem_, const char *field, void *value)
{
    sim_esdirk_memory *mem = mem_;

    if (!strcmp(field, "time_sim"))
    {
        double *ptr = value;
        *ptr = mem->time_sim;
    }
    else if (!strcmp(field, "time_sim_ad"))
    {
        double *ptr = value;
        *ptr = mem->time_ad;
    }
    else if (!strcmp(field, "time_sim_la"))
    {
        double *ptr = value;
        *ptr = mem->time_la;
    }
    else if (!strcmp(field, "cost_hess"))
    {
        struct blasfeo_dmat **ptr = value;
        *ptr = mem->cost_hess;
    }
    else if (!strcmp(field, "S_p"))
    {
        sim_esdirk_dims *dims = (sim_esdirk_dims *) dims_;

        if (dims->np == 0)
            return;

        if (mem->S_p == NULL)
        {
            printf("sim_esdirk_memory_get field %s requested but not allocated! Enable sens_forw_p.\n", field);
            exit(1);
        }

        blasfeo_unpack_dmat(dims->nx, dims->np, mem->S_p, 0, 0, value, dims->nx);
    }
    else
    {
        printf("sim_esdirk_memory_get field %s is not supported! \n", field);
        exit(1);
    }
}



/************************************************
 * workspace
 ************************************************/

acados_size_t sim_esdirk_workspace_calculate_size(void *config_, void *dims_, void *opts_)
{
    sim_esdirk_dims *dims = (sim_esdirk_dims *) dims_;
    sim_opts *opts = opts_;

    int ns = opts->ns;

    int nx = dims->nx;
    int nu = dims->nu;
    int nz = dims->nz;
    int ny = dims->ny;
    int nxz = (nx + nz);

    int steps = opts->num_steps;

    acados_size_t size = sizeof(sim_esdirk_workspace);

    /* blasfeo structs */
    // TODO(@anton) implement also non-reuse mode
    size += ns * sizeof(struct blasfeo_dvec); // rf
    size += ns * sizeof(struct blasfeo_dvec); // s
    size += ns * sizeof(struct blasfeo_dvec); // kz
    size += ns * sizeof(struct blasfeo_dvec); // dkz
    size += sizeof(struct blasfeo_dvec); // xn

    // TODO(@anton) maybe I only actually need one?
    size += ns * sizeof(struct blasfeo_dmat); // df_dkz

    /* blasfeo mem */
    size += ns * blasfeo_memsize_dvec(nx+nz);	// rf
    size += ns * blasfeo_memsize_dvec(nxz);	// xz
    size += ns * blasfeo_memsize_dvec(nxz);	// dkz
    size += ns * blasfeo_memsize_dvec(nx);	// s
    size += blasfeo_memsize_dvec(nx);		// xn

    size += 2 * blasfeo_memsize_dmat(nxz, nx);  // df_dx, df_dxdot
    size += blasfeo_memsize_dmat(nxz, nu);      // df_du
    size += blasfeo_memsize_dmat(nxz, nz);      // df_dz
    
    size += blasfeo_memsize_dmat(nxz, nxz);      // df_dkz

    size += nxz * ns * sizeof(int); // ipiv

    size += 1 * 8; // initial alignment
    make_int_multiple_of(64, &size);
    size += 1 * 64;

    return size;
}

static void *sim_esdirk_workspace_cast(void *config_, void *dims_, void *opts_, void *raw_memory)
{
    sim_opts *opts = opts_;
    sim_esdirk_dims *dims = (sim_esdirk_dims *) dims_;

    int ns = opts->ns;

    int nx = dims->nx;
    int nu = dims->nu;
    int nz = dims->nz;
    int ny = dims->ny;
    int nxz = (nx + nz);

    int steps = opts->num_steps;

    char *c_ptr = (char *) raw_memory;

    // initial align
    align_char_to(8, &c_ptr);

    sim_esdirk_workspace *workspace = (sim_esdirk_workspace *) c_ptr;
    c_ptr += sizeof(sim_esdirk_workspace);

    assign_and_advance_blasfeo_dvec_structs(ns, &workspace->rf, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(ns, &workspace->s, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(ns, &workspace->kz, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(ns, &workspace->dkz, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(1, workspace->xn, &c_ptr);

    /* algin c_ptr to 64 blasfeo_dmat_mem has to be assigned directly after that  */
    align_char_to(64, &c_ptr);

    assign_and_advance_blasfeo_dmat_mem(nxz, nx, &workspace->df_dx, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nxz, nx, &workspace->df_dxdot, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nxz, nu, &workspace->df_du, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nxz, nz, &workspace->df_dz, &c_ptr);

    for (int ii = 0; ii < ns; ii++) {
	      assign_and_advance_blasfeo_dvec_mem(nxz, &workspace->rf[ii], &c_ptr);
	      assign_and_advance_blasfeo_dvec_mem(nxz, &workspace->kz[ii], &c_ptr);
	      assign_and_advance_blasfeo_dvec_mem(nxz, &workspace->dkz[ii], &c_ptr);
	      assign_and_advance_blasfeo_dvec_mem(nx,  &workspace->s[ii], &c_ptr);
        assign_and_advance_blasfeo_dmat_mem(nxz, nxz, &workspace->df_dkz[ii], &c_ptr);
    }
    assign_and_advance_blasfeo_dvec_mem(nx, &workspace->xn, &c_ptr);

    assign_and_advance_int(nxz * ns, &workspace->ipiv, &c_ptr);

    // printf("\npointer moved - size calculated = %d bytes\n", c_ptr- (char*)raw_memory -
    // sim_esdirk_calculate_workspace_size(dims, opts_));

    assert((char *) raw_memory + sim_esdirk_workspace_calculate_size(config_, dims, opts_) >= c_ptr);

    return (void *) workspace;
}


size_t sim_esdirk_get_external_fun_workspace_requirement(void *config_, void *dims_, void *opts_, void *model_)
{
    esdirk_model *model = model_;

    size_t size = 0;
    size_t tmp_size;

    tmp_size = external_function_get_workspace_requirement_if_defined(model->impl_ode_fun);
    size = size > tmp_size ? size : tmp_size;
    tmp_size = external_function_get_workspace_requirement_if_defined(model->impl_ode_fun_jac_x_xdot_z);
    size = size > tmp_size ? size : tmp_size;
    tmp_size = external_function_get_workspace_requirement_if_defined(model->impl_ode_hess);
    size = size > tmp_size ? size : tmp_size;
    tmp_size = external_function_get_workspace_requirement_if_defined(model->impl_ode_jac_x_xdot_u_z);
    size = size > tmp_size ? size : tmp_size;
    tmp_size = external_function_get_workspace_requirement_if_defined(model->impl_dae_jac_p);
    size = size > tmp_size ? size : tmp_size;

    return size;
}


void sim_esdirk_set_external_fun_workspaces(void *config_, void *dims_, void *opts_, void *model_, void *workspace_)
{
    esdirk_model *model = model_;

    external_function_set_fun_workspace_if_defined(model->impl_ode_fun, workspace_);
    external_function_set_fun_workspace_if_defined(model->impl_ode_fun_jac_x_xdot_z, workspace_);
    external_function_set_fun_workspace_if_defined(model->impl_ode_hess, workspace_);
    external_function_set_fun_workspace_if_defined(model->impl_dae_jac_p, workspace_);
    external_function_set_fun_workspace_if_defined(model->impl_ode_jac_x_xdot_u_z, workspace_);
}


int sim_esdirk_precompute(void *config_, sim_in *in, sim_out *out, void *opts_, void *mem_,
                       void *work_)
{
    return ACADOS_SUCCESS;
}



void sim_esdirk_compute_z_and_algebraic_sens(sim_esdirk_dims *dims, sim_opts *opts, sim_in *in, sim_out *out, sim_esdirk_memory *mem, sim_esdirk_workspace *workspace, esdirk_model *model)
{
    /* int nx = dims->nx; */
    /* int nz = dims->nz; */
    /* int nu = dims->nu; */
    /* int ns = opts->ns; */

    /* acados_timer timer_ad, timer_la; */

    /* double timing_ad = 0.0; */
    /* double timing_la = 0.0; */

    /* double *u = in->u; */
    /* double t0 = in->t0; */

    /* double *Z_work = workspace->Z_work; */
    /* double *S_algebraic = out->S_algebraic; */
    /* struct blasfeo_dvec *K = workspace->K; */

    /* struct blasfeo_dmat *df_dx = &workspace->df_dx; */
    /* struct blasfeo_dmat *df_dxdot = &workspace->df_dxdot; */
    /* struct blasfeo_dmat *df_du = &workspace->df_du; */
    /* struct blasfeo_dmat *df_dz = &workspace->df_dz; */

    /* struct blasfeo_dmat *df_dxdotz = &workspace->df_dxdotz; */
    /* struct blasfeo_dmat *dk0_dxu = &workspace->dk0_dxu; */

    /* int *ipiv_one_stage = workspace->ipiv_one_stage; */
    /* struct blasfeo_dvec *xtdot = &workspace->xtdot; */

    /* struct blasfeo_dvec *rG = workspace->rG; */

    /* struct blasfeo_dmat *dK_dxu = workspace->dK_dxu; */
    /* struct blasfeo_dmat *dK_dxu_ss; */

    /* if (opts->sens_hess){ */
    /*     dK_dxu_ss = &dK_dxu[0]; */
    /* } */
    /* else */
    /* { */
    /*     dK_dxu_ss = dK_dxu; */
    /* } */

    /* if (opts->sens_algebraic && !opts->exact_z_output) */
    /* { */
    /*     double interpolated_value; */
    /*     for (int jj = 0; jj < nx+nu; jj++) */
    /*     { */
    /*         for (int ii = 0; ii < nz; ii++) */
    /*         { */
    /*             for (int kk = 0; kk < ns; kk++) */
    /*             { */
    /*                 Z_work[kk] = blasfeo_dgeex1(dK_dxu_ss, nx*ns+kk*nz+ii, jj); */
    /*             } */
    /*             neville_algorithm(0.0, ns - 1, opts->c_vec, Z_work, &interpolated_value); */
    /*             // eval polynomial through vals in Z_work at 0. */
    /*             S_algebraic[ii+jj*nz] = -interpolated_value; */
    /*         } */
    /*     } */
    /* } */

    /* if (opts->output_z || opts->sens_algebraic) */
    /* { */
    /*     for (int ii = 0; ii < nz; ii++) */
    /*     { */
    /*         for (int jj = 0; jj < ns; jj++) */
    /*         { */
    /*             Z_work[jj] = blasfeo_dvecex1(K, nx * ns + nz * jj + ii); */
    /*             // copy values of z_ii in first step, into Z_work */
    /*         } */
    /*         neville_algorithm(0.0, ns - 1, opts->c_vec, Z_work, &out->zn[ii]); */
    /*         // eval polynomial through (c_jj, z_jj) at 0. */
    /*     } */
    /* } */

    /* if (opts->exact_z_output) */
    /* { */
    /*     // INPUT: impl_ode */
    /*     ext_fun_arg_t impl_ode_type_in[5]; */
    /*     void *impl_ode_in[5]; */

    /*     // set input for impl_ode */
    /*     impl_ode_type_in[0] = COLMAJ; */
    /*     impl_ode_type_in[1] = BLASFEO_DVEC; */
    /*     impl_ode_type_in[2] = COLMAJ; */
    /*     impl_ode_type_in[3] = COLMAJ; */
    /*     impl_ode_type_in[4] = COLMAJ; */

    /*     impl_ode_in[0] = in->x; */
    /*     impl_ode_in[1] = xtdot; */
    /*     impl_ode_in[2] = u; */
    /*     impl_ode_in[3] = &out->zn[0]; */
    /*     impl_ode_in[4] = &t0; */

    /*     // impl_ode_fun_jac_x_xdot_z */
    /*     ext_fun_arg_t impl_ode_fun_jac_x_xdot_z_type_out[4]; */
    /*     struct blasfeo_dvec_args impl_ode_res_out; */
    /*     impl_ode_res_out.x = rG; */

    /*     void *impl_ode_fun_jac_x_xdot_z_out[4]; */
    /*     impl_ode_fun_jac_x_xdot_z_type_out[0] = BLASFEO_DVEC_ARGS; */
    /*     impl_ode_fun_jac_x_xdot_z_out[0] = &impl_ode_res_out; */
    /*     impl_ode_fun_jac_x_xdot_z_type_out[1] = BLASFEO_DMAT; */
    /*     impl_ode_fun_jac_x_xdot_z_out[1] = df_dx; */
    /*     impl_ode_fun_jac_x_xdot_z_type_out[2] = BLASFEO_DMAT; */
    /*     impl_ode_fun_jac_x_xdot_z_out[2] = df_dxdot; */
    /*     impl_ode_fun_jac_x_xdot_z_type_out[3] = BLASFEO_DMAT; */
    /*     impl_ode_fun_jac_x_xdot_z_out[3] = df_dz; */

    /*     // impl_ode_jac_x_xdot_u_z */
    /*     ext_fun_arg_t impl_ode_jac_x_xdot_u_z_type_out[4]; */
    /*     void *impl_ode_jac_x_xdot_u_z_out[4]; */
    /*     impl_ode_jac_x_xdot_u_z_type_out[0] = BLASFEO_DMAT; */
    /*     impl_ode_jac_x_xdot_u_z_out[0] = df_dx; */
    /*     impl_ode_jac_x_xdot_u_z_type_out[1] = BLASFEO_DMAT; */
    /*     impl_ode_jac_x_xdot_u_z_out[1] = df_dxdot; */
    /*     impl_ode_jac_x_xdot_u_z_type_out[2] = BLASFEO_DMAT; */
    /*     impl_ode_jac_x_xdot_u_z_out[2] = df_du; */
    /*     impl_ode_jac_x_xdot_u_z_type_out[3] = BLASFEO_DMAT; */
    /*     impl_ode_jac_x_xdot_u_z_out[3] = df_dz; */

    /*     if (opts->output_z || opts->sens_algebraic) */
    /*     { */
    /*         // initial guess for xdot0 */
    /*         for (int ii = 0; ii < nx; ii++) */
    /*         { */
    /*             double interpolated_value; */
    /*             for (int jj = 0; jj < ns; jj++) */
    /*             { */
    /*                 // copy values of k_ii in first step, into Z_work */
    /*                 Z_work[jj] = blasfeo_dvecex1(K, nx * jj + ii); */
    /*             } */
    /*             neville_algorithm(0.0, ns - 1, opts->c_vec, Z_work, &interpolated_value); */
    /*             // eval polynomial through (c_jj, k_jj) at 0. */
    /*             blasfeo_pack_dvec(1, &interpolated_value, 1, xtdot, ii); */
    /*         } */
    /*         // perform extra newton iterations to get xdot0, z0 more precisely. */
    /*         for (int ii = 0; ii < opts->newton_iter; ii++) */
    /*         { */
    /*             if (ii == 0 || !opts->jac_reuse) */
    /*             { */
    /*                 // eval jacobians at interpolated values */
    /*                 acados_tic(&timer_ad); */
    /*                 model->impl_ode_fun_jac_x_xdot_z->evaluate( */
    /*                     model->impl_ode_fun_jac_x_xdot_z, impl_ode_type_in, impl_ode_in, */
    /*                     impl_ode_fun_jac_x_xdot_z_type_out, impl_ode_fun_jac_x_xdot_z_out); */
    /*                 timing_ad += acados_toc(&timer_ad); */

    /*                 // set up df_dxdotz */
    /*                 blasfeo_dgecp(nx + nz, nx, df_dxdot, 0, 0, df_dxdotz, 0, 0); */
    /*                 blasfeo_dgecp(nx + nz, nz, df_dz,    0, 0, df_dxdotz, 0, nx); */
    /*                 // factorize */
    /*                 blasfeo_dgetrf_rp(nx + nz, nx + nz, df_dxdotz, 0, 0, df_dxdotz, 0, 0, */
    /*                                                                             ipiv_one_stage); */
    /*             } */

    /*             // permute rhs */
    /*             blasfeo_dvecpe(nx + nz, ipiv_one_stage, rG, 0); */
    /*             // backsolve */
    /*             blasfeo_dtrsv_lnu(nx + nz, df_dxdotz, 0, 0, rG, 0, rG, 0); */
    /*             blasfeo_dtrsv_unn(nx + nz, df_dxdotz, 0, 0, rG, 0, rG, 0); */

    /*             blasfeo_daxpy(nx, -1.0, rG, 0, xtdot, 0, xtdot, 0); */
    /*             blasfeo_dveccp(nx, rG, 0, xtdot, 0); */
    /*             blasfeo_unpack_dvec(nz, rG, nx, mem->z, 1); */
    /*             for (int jj = 0; jj < nz; jj++) */
    /*             { */
    /*                 out->zn[jj] -= mem->z[jj]; */
    /*             } */
    /*         } */
    /*     } */

    /*     if (opts->sens_algebraic) */
    /*     { */
    /*         /\* implicit function theorem to get S_alg *\/ */
    /*         // eval jacobians at interpolated values */
    /*         acados_tic(&timer_ad); */
    /*         model->impl_ode_jac_x_xdot_u_z->evaluate( */
    /*                 model->impl_ode_jac_x_xdot_u_z, impl_ode_type_in, impl_ode_in, */
    /*                 impl_ode_jac_x_xdot_u_z_type_out, impl_ode_jac_x_xdot_u_z_out); */
    /*         timing_ad += acados_toc(&timer_ad); */

    /*         // set up df_dxdotz */
    /*         blasfeo_dgecp(nx + nz, nx, df_dxdot, 0, 0, df_dxdotz, 0, 0); */
    /*         blasfeo_dgecp(nx + nz, nz, df_dz,    0, 0, df_dxdotz, 0, nx); */
    /*         // set up right hand side dk0_dxu */
    /*         blasfeo_dgecp(nx + nz, nx, df_dx, 0, 0, dk0_dxu, 0, 0); */
    /*         blasfeo_dgecp(nx + nz, nu, df_du, 0, 0, dk0_dxu, 0, nx); */

    /*         // solve linear system */
    /*         acados_tic(&timer_la); */
    /*         blasfeo_dgetrf_rp(nx + nz, nx + nz, df_dxdotz, 0, 0, df_dxdotz, 0, 0, ipiv_one_stage); */
    /*         blasfeo_drowpe(nx + nz, ipiv_one_stage, dk0_dxu); */
    /*         blasfeo_dtrsm_llnu(nx + nz, nx + nu, 1.0, df_dxdotz, 0, 0, */
    /*                         dk0_dxu, 0, 0, dk0_dxu, 0, 0); */
    /*         blasfeo_dtrsm_lunn(nx + nz, nx + nu, 1.0, df_dxdotz, 0, 0, */
    /*                         dk0_dxu, 0, 0, dk0_dxu, 0, 0); */
    /*         timing_la += acados_toc(&timer_la); */

    /*         // solution has different sign */
    /*         blasfeo_dgesc(nx + nz, nx + nu, -1.0, dk0_dxu, 0, 0); */

    /*         // extract output */
    /*         blasfeo_unpack_dmat(nz, nx + nu, dk0_dxu, nx, 0, S_algebraic, nz); */
    /*     } // if sens_algebraic */
    /* } // if exact_z_output */
    /* out->info->LAtime += timing_la; */
    /* out->info->ADtime += timing_ad; */
}


/************************************************
 * integrator
 ************************************************/

// helpers!
// TODO(@anton): extract helpers

// Main simulation
int sim_esdirk(void *config_, sim_in *in, sim_out *out, void *opts_, void *mem_, void *work_)
{
    acados_timer timer, timer_ad, timer_la;
    acados_tic(&timer);

    out->info->LAtime = 0.0;
    out->info->ADtime = 0.0;
    double timing_ad = 0.0;
    double timing_la = 0.0;

    // Get variables from workspace, etc;
    // cast pointers
    sim_config *config = config_;
    sim_opts *opts = opts_;

    if ( opts->ns != opts->tableau_size )
    {
        printf("Error in sim_esdirk: the Butcher tableau size does not match ns");
        exit(1);
    }
    int ns = opts->ns;

    void *dims_ = in->dims;
    sim_esdirk_dims *dims = (sim_esdirk_dims *) dims_;
    sim_esdirk_workspace *workspace =
        (sim_esdirk_workspace *) sim_esdirk_workspace_cast(config, dims, opts, work_);

    sim_esdirk_memory *mem = (sim_esdirk_memory *) mem_;

    esdirk_model *model = in->model;

    if (model->impl_ode_fun == 0)
    {
        printf("sim ESDIRK: impl_ode_fun is not provided. Exiting.\n");
        exit(1);
    }

    int nx = dims->nx;
    int nu = dims->nu;
    int nz = dims->nz;
    int ny = dims->ny;
    int np = dims->np;
    int nf_p = opts->sens_forw_p ? np : 0;
    int nxz = nx + nz;

    double *u = in->u;
    double t0 = in->t0;
    double t_current;

    int newton_iter = opts->newton_iter;
    double *A_mat = opts->A_mat;
    double *b_vec = opts->b_vec;
    int num_steps = opts->num_steps;
    double step = in->T / num_steps;

    int *ipiv = workspace->ipiv;

    struct blasfeo_dvec *rf = workspace->rf;
    struct blasfeo_dvec *kz = workspace->kz;
    struct blasfeo_dvec *dkz = workspace->dkz;
    struct blasfeo_dvec *s = workspace->s;
    struct blasfeo_dvec *xn = workspace->xn;
    struct blasfeo_dmat *df_dkz = workspace->df_dkz;

    struct blasfeo_dmat df_dx  = workspace->df_dx;
    struct blasfeo_dmat df_dxdot  = workspace->df_dxdot;
    struct blasfeo_dmat df_du  = workspace->df_du;
    struct blasfeo_dmat df_dz  = workspace->df_dz;

    /* if (nf_p > 0) */
    /* { */
    /*     if (model->impl_dae_jac_p == 0) */
    /*     { */
    /*         printf("sim ESDIRK: impl_dae_jac_p is not provided but sens_forw_p=true.\n"); */
    /*         exit(1); */
    /*     } */
    /*     blasfeo_dgese(nx, np, 0.0, S_p, 0, 0); */
    /* } */

    struct blasfeo_dvec_args impl_ode_x_in;
    struct blasfeo_dvec_args impl_ode_xdot_in;
    struct blasfeo_dvec_args impl_ode_z_in; impl_ode_z_in.xi = nx; // set offset in kz
    struct blasfeo_dvec_args impl_ode_res_out;
    // TODO(@anton): please let's move this into a function
    // SET FUNCTION IN- & OUTPUT TYPES
    // INPUT: impl_ode
    ACADOS_DEFINE_INOUT(impl_ode_in,5);
    ACADOS_SET_INOUT(impl_ode_in,	0,	BLASFEO_DVEC_ARGS,	&impl_ode_x_in);
    ACADOS_SET_INOUT(impl_ode_in,	1,	BLASFEO_DVEC_ARGS,	&impl_ode_xdot_in);
    ACADOS_SET_INOUT(impl_ode_in,	2,	COLMAJ,			u);
    ACADOS_SET_INOUT(impl_ode_in,	3,	BLASFEO_DVEC_ARGS,	&impl_ode_z_in);
    ACADOS_SET_INOUT(impl_ode_in,	4,	COLMAJ,			&t_current);
    // OUTPUT:
    // impl_ode_fun
    ACADOS_DEFINE_INOUT(impl_ode_fun_out,1);
    ACADOS_SET_INOUT(impl_ode_in, 0, BLASFEO_DVEC_ARGS, &impl_ode_res_out);
    impl_ode_res_out.x = rf;

    // impl_ode_fun_jac_x_xdot_z
    ACADOS_DEFINE_INOUT(impl_ode_fun_jac_x_xdot_z_out,4);
    ACADOS_SET_INOUT(impl_ode_fun_jac_x_xdot_z_out,	0,	BLASFEO_DVEC_ARGS,	&impl_ode_res_out);
    ACADOS_SET_INOUT(impl_ode_fun_jac_x_xdot_z_out,	1,	BLASFEO_DMAT,		&df_dx);
    ACADOS_SET_INOUT(impl_ode_fun_jac_x_xdot_z_out,	2,	BLASFEO_DMAT,		&df_dxdot);
    ACADOS_SET_INOUT(impl_ode_fun_jac_x_xdot_z_out,	3,	BLASFEO_DMAT,		&df_dz);

    // impl_ode_jac_x_xdot_u_z
    ACADOS_DEFINE_INOUT(impl_ode_jac_x_xdot_u_z_out,4);
    ACADOS_SET_INOUT(impl_ode_jac_x_xdot_u_z_out,	0,	BLASFEO_DMAT,	&df_dx);
    ACADOS_SET_INOUT(impl_ode_jac_x_xdot_u_z_out,	1,	BLASFEO_DMAT,	&df_dxdot);
    ACADOS_SET_INOUT(impl_ode_jac_x_xdot_u_z_out,	2,	BLASFEO_DMAT,	&df_du);
    ACADOS_SET_INOUT(impl_ode_jac_x_xdot_u_z_out,	3,	BLASFEO_DMAT,	&df_dz);

    // impl_dae_jac_p
    /* ACADOS_DEFINE_INOUT(impl_dae_jac_p_out,1); */
    /* ACADOS_SET_INOUT(impl_ode_jac_x_xdot_u_z_out, 0, BLASFEO_DMAT, &df_dp); */

 
    /* Initialize & Pack */
    // initialize
    // TODO(@anton) clear hessian

    // TODO(@anton) cost integration

    // pack initial state
    blasfeo_pack_dvec(nx, in->x, 1, xn, 0);
    // TODO(@anton) sensitivities
    // blasfeo_pack_dmat(nx, nx + nu, in->S_forw, nx, S_forw, 0, 0);
    // blasfeo_pack_dvec(nx + nu, in->S_adj, 1, lambda, 0);

    // initialize integration variables
    // TODO(@anton)
    for (int i = 0; i < ns; ++i)
    {
        // state derivatives
        blasfeo_pack_dvec(nx, mem->xdot, 1, kz+i, 0);
        // algebraic variables
        blasfeo_pack_dvec(nz, mem->z, 1, kz+i, nx);
    }

    // declare step pointers
    struct blasfeo_dmat *df_dkz_ss_ii; // df_dxz at step ss, stage ii
    struct blasfeo_dvec *rf_ss_ii; // df_dxz at step ss, stage ii
    int *ipiv_ss_ii;

		// element of A tableau
		double a;

    /************************************************
    * Forward Sweep
    *       - (simulation & forward sensitivities)
    ************************************************/
    // set input for forward sweep

    // start the loop
    for (int ss = 0; ss < num_steps; ss++)
    {
				// TODO(@anton) Handle case when we keep the whole trajectory
				df_dkz_ss_ii = workspace->df_dkz;
				rf_ss_ii = workspace->rf;
				ipiv_ss_ii = workspace->ipiv;
				// Do newton iterations on G
				for (int iter = 0; iter < newton_iter; iter++)
				{
						if ((opts->jac_reuse && (ss == 0) && (iter == 0)) || (!opts->jac_reuse))
						{
								// if new jacobian gets computed, initialize dG_dK_ss with zeros
								for (int ii = 0; ii < ns; ii++)
								{
										blasfeo_dgese(nxz, nxz, 0.0, df_dkz_ss_ii+ii, 0, 0);
								}
						}
						// set input vectors
						impl_ode_x_in.x = xn;
						impl_ode_xdot_in.x = kz;
						impl_ode_z_in.x = kz;
						// set the output vector
						impl_ode_res_out.x = rf_ss_ii;
						// build and factorize the matrices on the diagonal
						// TODO(@anton) make sure all is correct
						// first handle the first stage
						if ((opts->jac_reuse && (ss == 0) && (iter == 0)) || (!opts->jac_reuse))
						{   // evaluate the ode function & jacobian w.r.t. xn, xdot, z;
								// &  compute jacobian dG_dK_ss;

								acados_tic(&timer_ad);
								model->impl_ode_fun_jac_x_xdot_z->evaluate(
										model->impl_ode_fun_jac_x_xdot_z, impl_ode_in_type, impl_ode_in,
										impl_ode_fun_jac_x_xdot_z_out_type, impl_ode_fun_jac_x_xdot_z_out);
								timing_ad += acados_toc(&timer_ad);

								// TODO(@anton) is this copy necessary or can I replace the ode_res_out
								// copy into the work matrix and do LU factorization
								blasfeo_dgecp(nxz, nx, &df_dxdot, 0, 0, df_dkz_ss_ii, 0, 0);
								blasfeo_dgecp(nxz, nz, &df_dz, 0, 0, df_dkz_ss_ii, 0, nx);
								blasfeo_dgetrf_rp(nxz, nxz, df_dkz_ss_ii, 0, 0, df_dkz_ss_ii, 0, 0, ipiv_ss_ii);

						}
						else
						{
								// TODO(@anton) only recompute residual
						}
						// permute
						blasfeo_dvecpe(nxz, ipiv_ss_ii, rf_ss_ii, 0);
						// then backsolve storing the result in dkz
						blasfeo_dtrsv_lnu(nxz, df_dkz_ss_ii, 0, 0, rf_ss_ii, 0, dkz, 0);
						blasfeo_dtrsv_unn(nxz, df_dkz_ss_ii, 0, 0, dkz, 0, dkz, 0);
						// handle the rest of the stages
						for (int ii = 1; ii < ns; ii++)
            {  // ii-th row of tableau
								df_dkz_ss_ii = df_dkz+ii;
								rf_ss_ii = rf+ii;
								ipiv_ss_ii = ipiv+ii;
								// set input vectors
								impl_ode_x_in.x = s+ii;
								impl_ode_xdot_in.x = kz+ii;
								impl_ode_z_in.x = kz+ii;
								// set the output vector
								impl_ode_res_out.x = rf_ss_ii+ii;

                // take x(n); copy a strvec into a strvec
                blasfeo_dveccp(nx, xn, 0, s+ii, 0);
                t_current = t0 + ss * step + opts->c_vec[ii] * step;

                for (int jj = 0; jj < ii; jj++)
                {  // jj-th col of tableau
                    // Compute the new state
                    a = A_mat[ii + ns * jj] * step;
                    // xt = xt + T_int * a[i,j]*K_j
                    blasfeo_daxpy(nx, a, kz + jj, 0, s+ii, 0, s+ii, 0);
                }

								if ((opts->jac_reuse && (ss == 0) && (iter == 0)) || (!opts->jac_reuse))
								{
										acados_tic(&timer_ad);
										model->impl_ode_fun_jac_x_xdot_z->evaluate(
												model->impl_ode_fun_jac_x_xdot_z, impl_ode_in_type, impl_ode_in,
												impl_ode_fun_jac_x_xdot_z_out_type, impl_ode_fun_jac_x_xdot_z_out);
										timing_ad += acados_toc(&timer_ad);

										for (int jj = 0; jj < ii-1; jj++)
										{
												a = A_mat[ii + ns * jj] * step;
												// update the rf_ss_ii
												blasfeo_dgemv_n(nx+nz, nx, a, &df_dx, 0, 0, rf+jj, 0, 1.0, rf_ss_ii, 0, rf_ss_ii, 0);
										}

										a = A_mat[ii + ns * ii] * step;
										// TODO(@anton) is this copy necessary or can I replace the ode_res_out
										// copy into the work matrix and do LU factorization
										blasfeo_dgecp(nxz, nx, &df_dxdot, 0, 0, df_dkz_ss_ii, 0, 0);
										blasfeo_dgecp(nxz, nz, &df_dz, 0, 0, df_dkz_ss_ii, 0, nx);
										blasfeo_dgead(nxz, nx, a, &df_dx, 0, 0, df_dkz_ss_ii, 0, 0); // add the dx contribution
										blasfeo_dgetrf_rp(nxz, nxz, df_dkz_ss_ii, 0, 0, df_dkz_ss_ii, 0, 0, ipiv_ss_ii);
                }
                else // only eval function (without jacobian)
                {
                    acados_tic(&timer_ad);
                    model->impl_ode_fun->evaluate(model->impl_ode_fun, impl_ode_in_type,
                                                  impl_ode_in, impl_ode_fun_out_type,
                                                  impl_ode_fun_out);
                    timing_ad += acados_toc(&timer_ad);
                }
								// permute
								blasfeo_dvecpe(nxz, ipiv_ss_ii, rf_ss_ii, 0);
								// then backsolve storing the result in rf_ss_ii
								blasfeo_dtrsv_lnu(nxz, df_dkz_ss_ii, 0, 0, rf_ss_ii, 0, rf_ss_ii, 0);
								blasfeo_dtrsv_unn(nxz, df_dkz_ss_ii, 0, 0, rf_ss_ii, 0, rf_ss_ii, 0);
            }  // end ii
        } // end newton_iter

        // obtain x(n+1)
        for (int ii = 0; ii < ns; ii++){
            // xn += b_i * k_i
            blasfeo_daxpy(nx, step * b_vec[ii], kz+ii, 0, xn, 0, xn, 0);
        }

        // algebraic variables output and corresponding sensitivity propagation
        if (ss == 0 && nz > 0)
        {
            sim_esdirk_compute_z_and_algebraic_sens(dims, opts, in, out, mem, workspace, model);
        }

        if (ss == num_steps-1)
        {
            // store last xdot, z values for next initialization
            blasfeo_unpack_dvec(nx, kz+ns-1, 0, mem->xdot, 1);
            blasfeo_unpack_dvec(nz, kz+ns-1, nx, mem->z, 1);
        }
    }  // end step loop (ss)

    /* if (opts->cost_computation) */
    /* { */
    /*     // scale cost function value */
    /*     mem->cost_fun[0] *= cost_scaling; */
    /* } */

    // extract results from forward sweep to output
    /* blasfeo_unpack_dvec(nx, xn, 0, x_out, 1); */

    /* if  ( opts->sens_forw || opts->sens_hess ) */
    /*     blasfeo_unpack_dmat(nx, nx + nu, S_forw_ss, 0, 0, S_forw_out, nx); */

/*****************************************************************************
 * Backward Sweep
 *       - (adjoint sensitivities & hessian propagation)
 *       - hessian via symmetric forward-backward sweep
 *                    (see Algorithm 2 from Quirynen2016)
 *       - Quirynen2016: Symmetric Hessian propagation for lifted collocation integrators in direct optimal control
 *******************************************************************************/
    if ( opts->sens_adj  || opts->sens_hess )
    {
				// TODO(@anton)
    }  // end if ( opts->sens_adj  || opts->sens_hess )


    // extract output
    /* if  ( opts->sens_adj  || opts->sens_hess ) */
    /*     blasfeo_unpack_dvec(nx + nu, lambda, 0, S_adj_out, 1); */
    /* if  ( opts->sens_hess ) */
    /* { */
    /*     blasfeo_dtrtr_u(nu+nx, Hess, 0, 0, Hess, 0, 0); */
    /*     // printf("Hess = (ESDIRK) \n"); */
    /*     // blasfeo_print_exp_dmat(nx + nu, nx + nu, Hess, 0, 0); */
    /*     blasfeo_unpack_dmat(nx+nu, nx+nu, Hess, 0, 0, out->S_hess, nx + nu); */
    /* } */

    out->info->CPUtime = acados_toc(&timer);
    // note: this is the time for factorization and solving the linear systems
    out->info->LAtime += timing_la;
    out->info->ADtime += timing_ad;

    mem->time_sim = out->info->CPUtime;
    mem->time_ad = out->info->ADtime;
    mem->time_la = out->info->LAtime;

    return ACADOS_SUCCESS;
}



void sim_esdirk_config_initialize_default(void *config_)
{
    sim_config *config = config_;

    config->evaluate = &sim_esdirk;
    config->precompute = &sim_esdirk_precompute;
    config->opts_calculate_size = &sim_esdirk_opts_calculate_size;
    config->opts_assign = &sim_esdirk_opts_assign;
    config->opts_initialize_default = &sim_esdirk_opts_initialize_default;
    config->opts_update = &sim_esdirk_opts_update;
    config->opts_set = &sim_esdirk_opts_set;
    config->opts_get = &sim_esdirk_opts_get;
    config->memory_calculate_size = &sim_esdirk_memory_calculate_size;
    config->memory_assign = &sim_esdirk_memory_assign;
    config->memory_set = &sim_esdirk_memory_set;
    config->memory_set_to_zero = &sim_esdirk_memory_set_to_zero;
    config->memory_get = &sim_esdirk_memory_get;
    config->workspace_calculate_size = &sim_esdirk_workspace_calculate_size;
    config->get_external_fun_workspace_requirement = &sim_esdirk_get_external_fun_workspace_requirement;
    config->set_external_fun_workspaces = &sim_esdirk_set_external_fun_workspaces;
    config->model_calculate_size = &sim_esdirk_model_calculate_size;
    config->model_assign = &sim_esdirk_model_assign;
    config->model_set = &sim_esdirk_model_set;
    config->dims_calculate_size = &sim_esdirk_dims_calculate_size;
    config->dims_assign = &sim_esdirk_dims_assign;
    config->dims_set = &sim_esdirk_dims_set;
    config->dims_get = &sim_esdirk_dims_get;
    return;
}
