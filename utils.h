#ifndef LIBRESPONSE_UTILS_H_
#define LIBRESPONSE_UTILS_H_

/*!
 * @file
 *
 * Utility functions.
 */

#include <armadillo>
#include <vector>

/*!
 * A useful debugging macro.
 */
#define LINE std::cout                          \
    << "The line is: " << __LINE__              \
    << " in file: " << __FILE__                 \
    << std::endl;

static const std::string dots(77, '.');
static const std::string dashes(77, '-');
static const std::string equals(77, '=');
static const std::string lcarats(77, '<');
static const std::string rcarats(77, '>');

/*!
 * Repack an Armadillo matrix of shape \f$(a,i)\f$ into a vector of
 * shape \f$(a*i)\f$, where \f$a\f$ is the fast index.
 *
 * @param[out] &v matrix packed into a vector
 * @param[in] &m matrix to be packed
 */
void repack_matrix_to_vector(arma::vec &v, const arma::mat &m);

/*!
 * Repack an Armadillo vector of shape \f$(a*i)\f$, where \f$a\f$ is
 * the fast index, into an Armadillo matrix of shape \f$(a,i)\f$.
 *
 * @param[out] &m unpacked matrix
 * @param[in] &v vector to be unpacked
 */
void repack_vector_to_matrix(arma::mat &m, const arma::vec &v);

/*!
 * Calculate the root-mean-square deviation between two
 * identically-typed and identically-sized Armadillo containers,
 * element-wise.
 *
 * @param[in] &T_new Armadillo vector, matrix, or cube
 * @param[out] &T_old Armadillo vector, matrix, or cube
 *
 * @return rmsd value
 */
template <typename T>
double rmsd(const T &T_new, const T &T_old) {
    return sqrt(arma::accu(arma::pow((T_new - T_old), 2)));
}

/*!
 * Check for elementwise closeness between two
 * indentically-typed Armadillo objects.
 *
 * @param[in] &X first Armadillo object
 * @param[in] &Y second Armadillo object
 * @param[in] tol desired tolerance
 * @param[out] &current_norm current value of the maximum norm
 *
 * @return Whether or not the two Armadillo objects are "close" within
 * the given tolerance.
 */
template <typename T>
bool is_close(const T &X, const T &Y, double tol, double &current_norm)
{
    bool close(false);
    current_norm = arma::norm(X - Y, "inf");
    if (current_norm < tol)
    {
        close = true;
    }
    return close;
}

/*!
 * Concatenate or flatten a vector of Armadillo cubes into a
 * single cube.
 *
 * The matrix shape must be identical across all cubes.
 *
 * @param[in] &v vector of cubes to be concatenated
 *
 * @return A cube whose slices are in the same order as the input
 * vector of cubes.
 */
arma::cube concatenate_cubes(const std::vector<arma::cube> &v);

/*!
 * Fully transform a matrix from the AO basis to the MO basis:
 * \f$\mathbf{MO} = \mathbf{C}^{\dagger} * \mathbf{AO} * \mathbf{C}\f$
 *
 * @param[out] MO Transformed matrix in MO basis.
 * @param[in] AO Input matrix in AO basis.
 * @param[in] C MO coefficient matrix.
 */
void AO2MO(arma::mat& MO, const arma::mat& AO, const arma::mat& C);

/*!
 * Transform a matrix from the AO basis to part of the MO basis: \f$
 * \mathbf{MO} = \mathbf{C}_{\mathrm{from}}^{\dagger} * \mathbf{AO} *
 * \mathbf{C}_{\mathrm{to}} \f$
 *
 * @param[out] MO Transformed matrix in partial MO basis.
 * @param[in] AO Input matrix in AO basis.
 * @param[in] C_from MO coefficient matrix in basis transforming from.
 * @param[in] C_to MO coefficient matrix in basis transforming to.
 */
void AO2MO(arma::mat& MO, const arma::mat& AO, const arma::mat& C_from, const arma::mat& C_to);

/*!
 * Flip the sign of the lower triangle of an Armadillo matrix
 * in-place.
 *
 * @param[in,out] &mat matrix to modify in-place
 */
void skew_lower(arma::mat &mat);

/*!
 * Flip the sign of the lower triangle of each slice of an
 * Armadillo cube in-place.
 *
 * @param[in,out] &cube cube to modify in-place
 */
void skew_lower(arma::cube &cube);

/*!
 * Flip the sign of the upper triangle of an Armadillo matrix
 * in-place.
 *
 * @param[in,out] &mat matrix to modify in-place
 */
void skew_upper(arma::mat &mat);

/*!
 * Flip the sign of the upper triangle of each slice of an
 * Armadillo cube in-place.
 *
 * @param[in,out] &cube cube to modify in-place
 */
void skew_upper(arma::cube &cube);

/*!
 * Check for matrix (anti,un)symmetry.
 *
 * This function returns
 * 1 if the matrix is symmetric to threshold THRZER
 * 2 if the matrix is antisymmetric to threshold THRZER
 * 3 if all elements are below THRZER
 * 0 otherwise (the matrix is unsymmetric about the diagonal)
 *
 * @param[in] &amat Armadillo matrix being tested for (anti)symmetry
 *
 * @return 1, 2, 3, or 0
 *
 * @sa DALTON/gp/gphjj.F/MATSYM, DALTON/include/thrzer.h
 */
int matsym(const arma::mat &amat);

/*!
 * Print a matrix to stdout with high precision.
 *
 * @param[in] &results matrix to be printed (each element is 24.18lf)
 */
void print_results_raw(const arma::mat &results);

/*!
 * Print a matrix to stdout with high precision, along with a label for each element.
 *
 * The number of labels must equal each dimension of the matrix.
 *
 * @param[in] &results matrix to be printed (each element is 24.18lf)
 * @param[in] &labels strings to go along with each element
 */
void print_results_with_labels(
    const arma::mat &results, const std::vector<std::string> &labels);

/*!
 * Print a matrix to stdout with high precision, along with a pair of
 * labels for each element.
 *
 * The number of labels must equal each dimension of the matrix.
 *
 * @param[in] &results matrix to be printed (each element is 24.18lf)
 * @param[in] &labels_1 strings to go along with each element
 * @param[in] &labels_2 strings to go along with each element
 */
void print_results_with_labels(
    const arma::mat &results,
    const std::vector<std::string> &labels_1,
    const std::vector<std::string> &labels_2
    );

/*!
 * Give the correct prefactor for the B matrix depending on
 * whether ot not the perturbing integrals are imaginary or real.
 *
 * The form of the RPA equations is \f$(\mathbf{A}+\mathbf{B})\f$ for
 * a real operator on the right-hand side and
 * \f$(\mathbf{A}-\mathbf{B})\f$ for an imaginary operator.
 *
 * @param[in] is_imaginary boolean for when or not we want the prefactor for an imaginary matrix
 *
 * @return -1 or 1 for imaginary/real
 */
int is_imaginary_to_b_prefactor(bool is_imaginary);

/*!
 * Combine two vectors into one.
 *
 * The total length of vc must be equal to the sum of the lengths of
 * v1 and v2.
 *
 * @param[out] &vc combined vector
 * @param[in] &v1 first vector
 * @param[in] &v2 second vector
 */
void join_vector(arma::vec &vc, const arma::vec &v1, const arma::vec &v2);

/*!
 * Split one vector into two.
 *
 * The total length of vc must be equal to the sum of the lengths of
 * v1 and v2.
 *
 * @param[in] &vc combined vector
 * @param[out] &v1 first vector
 * @param[out] &v2 second vector
 */
void split_vector(const arma::vec &vc, arma::vec &v1, arma::vec &v2);

/*!
 * Convert the strings "true" and "false" to the respective
 * boolean-type values.
 *
 * The string isn't passed by reference due to an internal
 * transformation to lowercase.
 * 
 * Taken from http://stackoverflow.com/a/3613424/3249688
 *
 * @param[in] str string to convert
 *
 * @returns bool (true/false)
 */
bool string_to_bool(std::string str);

/*!
 * Convert a boolean to the string "true" or "false".
 *
 * @param[in] b boolean
 *
 * @returns "true" or "false"
 */
std::string bool_to_string(bool b);

/*!
 * Transform a string entirely to uppercase.
 *
 * @param[in] s string to transform to uppercase
 *
 * @returns input string transformed entirely to uppercase
 */
std::string to_upper(std::string s);

/*!
 * Transform a string entirely to lowercase.
 *
 * @param[in] s string to transform to lowercase
 *
 * @returns input string transformed entirely to lowercase
 */
std::string to_lower(std::string s);

void pretty_print(const arma::mat &M, std::string title = "", bool sci = false, size_t width = 10, size_t numCols = 6);
void pretty_print(const arma::cube &C, std::string title = "", bool sci = false, size_t width = 10, size_t numCols = 6);

arma::uvec range(int start, int stop, int step);

arma::uvec range(int start, int stop);

arma::uvec range(int stop);

// How to enforce that we expect T to be some kind of Armadillo
// vector?
template<typename T>
T join(const T &a1, const T &a2)
{

    const size_t l1 = a1.n_elem;
    const size_t l2 = a2.n_elem;
    const size_t l3 = l1 + l2;

    T a3(l3);

    a3.subvec(0, l1 - 1) = a1;
    a3.subvec(l1, l3 - 1) = a2;

    return a3;

}

void print_polarizability(std::ostringstream &os, const arma::mat &polar_tensor);

void print_square_result(std::ostringstream &os, const arma::mat &square_result);

void printf_14_7_3by3(const arma::mat &m);

void printf_14_7_3by3_orientation(const arma::mat &m_ori);

void printf_14_7_row(const arma::vec &v);

#endif // LIBRESPONSE_UTILS_H_
