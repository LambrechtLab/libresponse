#include "configurable.h"
#include "utils.h"

void set_defaults(libresponse::configurable &options)
{
    options.cfg("order", "linear");
    options.cfg("solver", "diis");
    options.cfg("hamiltonian", "rpa");
    options.cfg("spin", "singlet");
    options.cfg<unsigned>("maxiter", 60);
    options.cfg<int>("conv", 8);
    options.cfg<unsigned>("diis_start", 1);
    options.cfg<unsigned>("diis_vectors", 7);
    options.cfg<bool>("rhf_as_uhf", false);
    options.cfg<int>("print_level", 2);
    options.cfg<int>("memory", 2000);
    options.cfg("integral_engine", "libint");
    options.cfg("run_type", "single");
    options.cfg<int>("save", 0);
    options.cfg<bool>("read", false);
    options.cfg<bool>("dump_ao_integrals", false);
    options.cfg<bool>("force_not_nonorthogonal", false);
    options.cfg<bool>("force_nonorthogonal", false);

    // If any of these are true, zero out the corresponding AO or MO
    // (ov/vo) interfragment matrix elements at that step. The ones
    // for AOs shouldn't be changed unless you know what you're doing!
    // The order here generally follows how they appear in the
    // implementation.
    options.cfg<bool>("_mask_operator_ao", false);
    options.cfg<bool>("_mask_rhsvec_mo", false);
    options.cfg<bool>("_mask_rspvec_guess_mo", false);
    options.cfg<bool>("_mask_dg_ao", false);
    options.cfg<bool>("_mask_j_ao", false);
    options.cfg<bool>("_mask_k_ao", false);
    options.cfg<bool>("_mask_ints_mnov_ao", false);
    options.cfg<bool>("_mask_product_mo", false);
    options.cfg<bool>("_mask_ediff_mo", false);
    options.cfg<bool>("_mask_rspvec_mo", false);
    options.cfg<bool>("_mask_form_results_mo", false);
    // When doing fragment-specific occ, full response, what fragment
    // to run response for.
    options.cfg<int>("_frgm_response_idx", 0);

    // TODO document me!
    options.cfg<bool>("_do_compute_generalized_density", true);
    options.cfg<bool>("_do_orthogonalization_canonical", false);
}
