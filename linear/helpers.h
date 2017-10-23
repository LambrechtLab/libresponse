#ifndef LIBRESPONSE_LINEAR_HELPERS_H_
#define LIBRESPONSE_LINEAR_HELPERS_H_

/*!
 * @file
 *
 * Core routines called by the response solvers.
 */

#include "../indices.h"
#include "../operator_spec.h"

namespace libresponse {

/*!
 * Given AO-basis integrals, transform them to vector-packed occ-virt subspace MO-basis integrals.
 *
 * First, performs the AO to MO transformation. \f$ i \f$ is an occupied MO index, \f$ a \f$ is a virtual MO index. \f$ M_{ai} = C_{\mu a} M_{\mu\nu} C_{\nu i} \f$.
 *
 * Second, repacks each transformed \f$ [a, i] \f$ matrix into an \f$ [(ia)] \f$ compound index vector, where \f$ a \f$ is the fast index.
 *
 * @param[out] &ia_vecs matrix of vectors of occ-virt MO integrals
 * @param[in] &mn_mats cube of AO integrals
 * @param[in] &C_occ MO coefficients, occupied subspace (1 spin)
 * @param[in] &C_virt MO coefficients, virtual subspace (1 spin)
 */
void one_electron_mn_mats_to_ia_vecs(
    arma::mat &ia_vecs,
    const arma::cube &mn_mats,
    const arma::mat &C_occ,
    const arma::mat &C_virt
    );

void one_electron_ia_vecs_to_mn_mats(
    arma::cube &mn_mats,
    const arma::mat &ia_vecs,
    const arma::mat &C_occ,
    const arma::mat &C_virt);

void one_electron_mn_mats_to_ia_vecs(
    arma::mat &ia_vecs,
    const arma::cube &mn_mats,
    const arma::mat &C_occ,
    const arma::mat &C_virt,
    const type::indices &mask_indices
    );

/*!
 * Form a vector of virtual-occupied MO energy differences using a compound index \f$ia\f$, where \f$a\f$ is the fast index.
 *
 * \f$ \Delta_{(ia)} = \epsilon_{a} - \epsilon_{i} \f$
 *
 * @param[out] &ediff Armadillo vector of energy differences
 * @param[in] &moene_i energies of occupied MOs
 * @param[in] &moene_a energies of virtual/unoccupied MOs
 */
void form_vec_energy_differences(
    arma::vec &ediff,
    const arma::vec &moene_i,
    const arma::vec &moene_a
    );

/*!
 * Compute the generalized density: \f$ D_{\mu\nu}^{X}[q_{bj}] = C_{\mu b} q_{bj}^{X} C_{\nu j} \f$.
 *
 * The index \f$j\f$ runs over occupied MOs and the index \f$b\f$ runs over unoccupied MOs. For the compound index \f$\{bj\}\f$, \f$b\f$ is fast and \f$j\f$ is slow.
 *
 * @param[out] &Dg generalized density
 * @param[in] &q occ-virt subspace MO basis vector (packed)
 * @param[in] &C_occ occupied subspace MO coefficients
 * @param[in] &C_virt virtual subspace MO coefficients
 */
void compute_generalized_density(
    arma::mat &Dg,
    const arma::vec &q,
    const arma::mat &C_occ,
    const arma::mat &C_virt
    );

/*!
 * Form a guess for the response vector from the gradient/RHS vector and the denominator of virt-occ MO energy differences.
 *
 * \f$ q_{ia} = \frac{V_{ia}{\Delta_{ia} - \omega} \f$
 *
 * For the compound index \f$\{ia\}/\{ai\}\f$, \f$a\f$ is fast and \f$i\f$ is slow.
 *
 * @param[out] &rspvec initial guess response vector
 * @param[in] &rhsvec packed vector of occ-virt gradient/RHS integrals
 * @param[in] &ediff vector of virt-occ MO energy differences
 * @param[in] frequency frequency of applied field in atomic units
 */
void form_guess_rspvec(
    arma::vec &rspvec,
    const arma::vec &rhsvec,
    const arma::vec &ediff,
    double frequency
    );

void form_guess_rspvec(
    arma::vec &rspvec,
    const arma::vec &rhsvec,
    const arma::mat &ediff,
    double frequency
    );

/*!
 * Update the response vector from the previous iteration's matrix-vector product, the gradient/RHS vector, and the denominator of virt-occ MO energy differences.
 *
 * \f$ q_{ia} = \frac{V_{ia} - G_{ia}}{\Delta_{ia} - \omega} \f$
 *
 * For the compound index \f$\{ia\}/\{ai\}\f$, \f$a\f$ is fast and \f$i\f$ is slow.
 *
 * @param[out] &rspvec updated response vector
 * @param[in] &product orbital Hessian-guess vector product
 * @param[in] &rhsvec packed vector of occ-virt gradient/RHS integrals
 * @param[in] &ediff vector of virt-occ MO energy differences
 * @param[in] frequency frequency of applied field in atomic units
 */
void form_new_rspvec(
    arma::vec &rspvec,
    const arma::vec &product,
    const arma::vec &rhsvec,
    const arma::vec &ediff,
    double frequency
    );

void form_new_rspvec(
    arma::vec &rspvec,
    const arma::vec &product,
    const arma::vec &rhsvec,
    const arma::mat &ediff,
    double frequency
    );

/*!
 * Contract each property vector with each response vector to form the final linear response values.
 *
 * @param[out] &results matrix of final LR values
 * @param[in] &vecs_property property vectors (each column)
 * @param[in] &vecs_response response vectors (each column)
 */
void form_results(
    arma::mat &results,
    const arma::mat &vecs_property,
    const arma::mat &vecs_response
    );

void form_results(
    arma::mat &results,
    const arma::mat &vecs_property,
    const arma::mat &vecs_response,
    const arma::uvec &indices_mo
    );

/*!
 * Contract all property vectors (for all operators) with all response vectors (for all operators) to form the final linear response values.
 *
 * The length of each std::vector is the number of operators.
 *
 * All property/response vectors for an operator are grouped
 * together in a matrix, and individual property/response vectors
 * for each component or atom are individual columns in that
 * matrix.
 *
 * @param[out] &results matrix of final LR values
 * @param[in] &vecs_property property vectors (each matrix an operator, each matrix column an operator component)
 * @param[in] &vecs_response response vectors (each matrix an operator, each matrix column an operator component)
 */
void form_results(
    arma::mat &results,
    const std::vector<arma::mat> &vecs_property,
    const std::vector<arma::mat> &vecs_response
    );

void form_results(
    arma::mat &results,
    const std::vector<arma::mat> &vecs_property,
    const std::vector<arma::mat> &vecs_response,
    const arma::uvec &indices_mo
    );

void form_results(
    arma::cube &results,
    const std::vector<operator_spec> &operators,
    const type::indices * indices_mo = NULL
    );

/*!
 * Convert MO occupation numbers (occupied and virtual, alpha and beta) to explicit ranges.
 *
 * Set the elements of ranges based upon the number of occupied and virtual orbital, and assume that all orbitals in each subspace are included.
 *
 * @param[in] &occupations 4-element vector of ranges (noa, nva, nob, nvb)
 *
 * @returns 4x2 matrix of boundaries
 */
arma::umat occupations_to_ranges(const arma::uvec &occupations);

arma::umat pack_fragment_occupations(
    const arma::uvec &nbasis_frgm,
    const arma::uvec &norb_frgm,
    const arma::uvec &nocc_frgm_alph,
    const arma::uvec &nocc_frgm_beta
    );

arma::cube check_fragment_locality(
    const arma::cube &mocoeffs,
    const arma::umat &fragment_occupations);

arma::cube weight_to_pct(const arma::cube &weights);

/*!
 * Form the half-transformed orbital Hessian \f$ G_{\mu\nu} \equiv A_{\mu\nu}\f$, \f$ (A+B)_{\mu\nu} \f$, or \f$ (A-B)_{\mu\nu} \f$ from \f$ J_{\mu\nu}^{X} \f$ and \f$ K_{\mu\nu}^{X} \f$.
 *
 * To form the fully-transformed orbital Hessian \f$ G_{ia} \f$,
 * the result will need to be transformed once more into the
 * occupied-virtual MO subspace.
 *
 * In theory, this could be used with any \f$ \mathbf{J} \f$ or
 * \f$ \mathbf{K} \f$, but isn't meaningful in other contexts.
 *
 * @todo Put the explicit equations somewhere in the documentation.
 *
 * @param[out] &orbhess orbital Hessian, shape \f$ [\mu,\nu] \f$
 * @param[in] &J generalized Coulomb matrices, shape \f$ [\mu,\nu] \f$, precontracted with \f$ D_{\mu\nu}[X_{jb}] \f$
 * @param[in] &K generalized exchange matrices, shape \f$ [\mu,\nu] \f$, precontracted with \f$ D_{\mu\nu}[X_{jb}] \f$
 * @param[in] &hamiltonian "rpa" or "tda"
 * @param[in] &spin "singlet" or "triplet"
 * @param[in] b_prefactor 1 or -1 (relevant for RPA, not TDA)
 */
void form_orbital_hessian_equations(
    arma::cube &orbhess,
    const arma::cube &J,
    const arma::cube &K,
    const std::string &hamiltonian,
    const std::string &spin,
    int b_prefactor
    );

void test_idempotency(const arma::mat &M, const arma::mat &S);

void form_ediff_terms(
    arma::mat &ediff_mat,
    const arma::mat &F,
    const arma::mat &S,
    size_t nocc,
    size_t nvirt
    );

void form_superoverlap(
    arma::mat &superoverlap,
    const arma::mat &S,
    size_t nocc,
    size_t nvirt
    );

}

#endif // LIBRESPONSE_LINEAR_HELPERS_H_
