#ifndef LIBRESPONSE_LINEAR_INTERFACE_NONORTHOGONAL_H_
#define LIBRESPONSE_LINEAR_INTERFACE_NONORTHOGONAL_H_

/*!
 * @file
 *
 *
 */

#include <armadillo>
#include <vector>
#include "../configurable.h"
#include "../matvec_i.h"
#include "../operator_spec.h"
#include "iterator.h"

namespace libresponse {

void solve_linear_response(
    arma::cube &results,
    MatVec_i *matvec,
    SolverIterator_nonorthogonal *solver_iterator,
    const arma::cube &C,
    const arma::umat &fragment_occupations,
    const arma::uvec &occupations,
    const arma::cube &F,
    const arma::mat &S,
    const std::vector<double> &omega,
    std::vector<operator_spec> &operators,
    const configurable &cfg
    );

} // namespace libresponse

#endif
