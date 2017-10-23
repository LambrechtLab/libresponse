#include <cassert>

#include "matvec_i.h"

MatVec_i::MatVec_i() { }
MatVec_i::~MatVec_i() { }

void MatVec_i::compute(arma::cube &J, arma::cube &K, arma::cube &P) { }

void MatVec_i::compute(arma::cube &J, arma::cube &K, const std::vector<arma::mat> &L, const std::vector<arma::mat> &R)
{

    // TODO I don't care about being efficient with memory/copying
    // here for now because this implementation is only *needed* for
    // Psi4.
    assert(L.size() == R.size());
    assert(L.size() == K.n_slices);
    const size_t nden = L.size();

    if (R[0].n_rows != L[0].n_rows)
        throw std::runtime_error("R[0].n_rows != L[0].n_rows");
    if (R[0].n_cols != L[0].n_cols)
        throw std::runtime_error("R[0].n_cols != L[0].n_cols");
    if (nden == 2) {
        if (R[1].n_rows != L[1].n_rows)
            throw std::runtime_error("R[1].n_rows != L[1].n_rows");
        if (R[1].n_cols != L[1].n_cols)
            throw std::runtime_error("R[1].n_cols != L[1].n_cols");
    }

    arma::cube P(J.n_rows, J.n_slices, nden);
    P.slice(0) = L[0] * R[0].t();
    if (nden == 2)
        P.slice(1) = L[1] * R[1].t();

    compute(J, K, P);

    return;

}
