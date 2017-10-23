#ifndef LIBRESPONSE_MATVEC_H_
#define LIBRESPONSE_MATVEC_H_

/*!
 * @file
 *
 * J/K integral generation: base class.
 */

#include <armadillo>

/*!
 * J/K integral generation: Base class (should only use derived classes).
 */
class MatVec_i {

public:

    MatVec_i();          //!< default constructor
    virtual ~MatVec_i(); //!< default destructor

    /*!
     * Compute J and K from P, where P is not necessarily symmetric.
     *
     * All cubes will either have 1 slice (for restricted
     * wavefunctions) or 2 slices (for unrestricted wavefunctions).
     *
     * @todo In the future, as long as the number of slices is
     * identical/consistent between J/K/P, this should handle an
     * arbitrary number.
     *
     * @param[out] &J generalized Coulomb matrices
     * @param[out] &K generalized exchange matrices
     * @param[in] &P generalized densities
     */
    virtual void compute(arma::cube &J, arma::cube &K, arma::cube &P);

    /*!
     * Compute J and K from C_left/L and C_right/R, where the
     * densities P formed from L/R are not necessarily symmetric.
     *
     * All cubes will either have 1 slice (for restricted
     * wavefunctions) or 2 slices (for unrestricted wavefunctions).
     *
     * @todo In the future, as long as the number of slices is
     * identical/consistent between J/K/L/R, this should handle an
     * arbitrary number.
     *
     * @param[out] &J generalized Coulomb matrices
     * @param[out] &K generalized exchange matrices
     * @param[in] &L generalized left MO-type coefficients
     * @param[in] &R generalized right MO-type coefficients
     */
    virtual void compute(arma::cube &J, arma::cube &K, const std::vector<arma::mat> &L, const std::vector<arma::mat> &R);

protected:

private:

};

#endif // LIBRESPONSE_MATVEC_H_
