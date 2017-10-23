#ifndef LIBRESPONSE_LINEAR_INTERFACE_H_
#define LIBRESPONSE_LINEAR_INTERFACE_H_

/*!
 * @file
 *
 * Function interfaces to linear response solvers.
 */

#include <armadillo>
#include <vector>
#include "../configurable.h"
#include "../matvec_i.h"
#include "../operator_spec.h"
#include "iterator.h"

namespace libresponse {

/*!
 * Solve the linear response equations for \f$\left\langle\left\langle \hat{V};\hat{W} \right\rangle\right\rangle_\omega \f$.
 *
 * \f$\hat{V}\f$ and \f$\hat{W}\f$ are taken to be from the same
 * list of operators, and all permutations are automatically
 * formed.
 *
 * To force treating a restricted reference as unrestricted,
 * duplicate the MO coefficients in a second cube slice and the MO
 * energies in a second matrix column.
 *
 * @param[out] &results Linear response values for all possible V and W operators, one slice per frequency.
 * @param[in] *matvec Two-electron integral computation object.
 * @param[in] &C MO coeffcients, 1 slice per alpha/beta spin
 * @param[in] &moene energies of all MOs, 1 column per alpha/beta spin
 * @param[in] &occupations 4 elements: nocc_alpha, nvirt_alpha, nocc_beta, nvirt_beta; if RHF, pass identical values for alpha and beta
 * @param[in] &omega One or more field frequencies in atomic units.
 * @param[in] &operators One or more operators to find LR values for.
 * @param[in] &cfg Map to hold configuration for solver
 */
void solve_linear_response(
    arma::cube &results,
    MatVec_i *matvec,
    SolverIterator_i<arma::vec> *solver_iterator,
    const arma::cube &C,
    const arma::mat &moene,
    const arma::uvec &occupations,
    const std::vector<double> &omega,
    std::vector<operator_spec> &operators,
    const configurable &cfg
    );

} // namespace libresponse

#endif // LIBRESPONSE_LINEAR_INTERFACE_H_
