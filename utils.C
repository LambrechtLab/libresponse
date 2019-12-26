#include <cassert>
#include <stdexcept>
#include "utils.h"

// For to_bool().
#include <sstream>
#include <string>
#include <iomanip>
#include <algorithm>
#include <cctype>

void repack_matrix_to_vector(arma::vec &v, const arma::mat &m)
{

    const size_t d1 = m.n_cols;
    const size_t d2 = m.n_rows;
    const size_t dv = d1 * d2;
    assert(v.n_elem == dv);

    // If not copying the memory, this is how to do it safely.
    // size_t ia;
    // for (size_t i = 0; i < d1; i++) {
    //     for (size_t a = 0; a < d2; a++) {
    //         ia = i*d2 + a;
    //         v(ia) = m(a, i);
    //     }
    // }

    memcpy(v.memptr(), m.memptr(), dv * sizeof(double));

    return;

}

void repack_vector_to_matrix(arma::mat &m, const arma::vec &v)
{

    const size_t d1 = m.n_cols;
    const size_t d2 = m.n_rows;
    const size_t dv = d1 * d2;
    assert(v.n_elem == dv);

    // If not copying the memory, this is how to do it safely.
    // size_t ia;
    // for (size_t i = 0; i < d1; i++) {
    //     for (size_t a = 0; a < d2; a++) {
    //         ia = i*d2 + a;
    //         m(a, i) = v(ia);
    //     }
    // }

    memcpy(m.memptr(), v.memptr(), dv * sizeof(double));

    return;
}

// TODO Do these need Doxygen comments, or is the template definition
// enough?
template double rmsd<arma::vec>(const arma::vec &T_new, const arma::vec &T_old);
template double rmsd<arma::mat>(const arma::mat &T_new, const arma::mat &T_old);

template bool is_close<arma::vec>(const arma::vec &X, const arma::vec &Y, double tol, double &current_norm);
template bool is_close<arma::mat>(const arma::mat &X, const arma::mat &Y, double tol, double &current_norm);

arma::cube concatenate_cubes(const std::vector<arma::cube> &v)
{

    const size_t n_cubes = v.size();

    if (n_cubes == 0) {
        throw 1;
    } else if (n_cubes == 1) {
        return v[0];
    } else {

        // Do some size checking.
        const size_t n_rows = v[0].n_rows;
        const size_t n_cols = v[0].n_cols;
        for (size_t iv = 0; iv < n_cubes; iv++) {
            if (v[iv].n_rows != n_rows) {
                std::cout << "n_rows inconsistent in concatenate_cubes" << std::endl;
                throw 1;
            }
            if (v[iv].n_cols != n_cols) {
                std::cout << "n_cols inconsistent in concatenate_cubes" << std::endl;
                throw 1;
            }
        }

        // Figure out how many slices in total are needed.
        size_t n_slices_total = 0;
        for (size_t iv = 0; iv < n_cubes; iv++) {
            n_slices_total += v[iv].n_slices;
        }

        arma::cube c(n_rows, n_cols, n_slices_total);

        // Place the original cubes in the right slices for the final
        // cube.
        size_t idx_slice_start = 0;
        for (size_t iv = 0; iv < n_cubes; iv++) {
            c.slices(idx_slice_start, idx_slice_start + v[iv].n_slices - 1) = v[iv];
            idx_slice_start += v[iv].n_slices;
        }

        return c;

    }

}

void AO2MO(arma::mat& MO, const arma::mat& AO, const arma::mat& C)
{

    MO = C.t() * AO * C;

    return;

}

void AO2MO(arma::mat& MO, const arma::mat& AO, const arma::mat& C_from, const arma::mat& C_to)
{

    MO = C_from.t() * AO * C_to;

    return;

}

void skew_lower(arma::mat& mat)
{

    mat = arma::trimatu(mat) - arma::trimatl(mat) - arma::diagmat(mat);

    return;

}


void skew_lower(arma::cube &cube)
{

    for (size_t s = 0; s < cube.n_slices; s++)
        skew_lower(cube.slice(s));

    return;

}

void skew_upper(arma::mat& mat)
{

    mat = arma::trimatl(mat) - arma::trimatu(mat) - arma::diagmat(mat);

    return;

}

void skew_upper(arma::cube &cube)
{

    for (size_t s = 0; s < cube.n_slices; s++)
        skew_upper(cube.slice(s));

    return;

}

int matsym(const arma::mat &amat)
{

    const double thrzer = 1.0e-14;

    assert(amat.n_rows == amat.n_cols);

    const size_t n = amat.n_rows;

    int isym = 1;
    int iasym = 2;

    double amats, amata;

    for (size_t j = 0; j < n; j++) {
        // The +1 is so the diagonal elements are checked.
        for (size_t i = 0; i < j+1; i++) {
            amats = std::abs(amat(i, j) + amat(j, i));
            amata = std::abs(amat(i, j) - amat(j, i));
            if (amats > thrzer)
                iasym = 0;
            if (amata > thrzer)
                isym = 0;
        }
    }

    return isym + iasym;

}

void print_results_raw(const arma::mat &results)
{

    for (size_t i = 0; i < results.n_rows; i++)
        for (size_t j = 0; j < results.n_cols; j++)
            std::cout << std::fixed << std::right
                      << "  [ " << std::setw(3) << i + 1 << " , " << j + 1 << " ]: "
                      << std::setw(24) << std::setprecision(18) << results(i, j) << std::endl;

    return;

}

void print_results_with_labels(
    const arma::mat &results,
    const std::vector<std::string> &labels) {

    assert(results.n_rows == labels.size());
    assert(results.n_cols == labels.size());

    for (size_t i = 0; i < results.n_rows; i++)
        for (size_t j = 0; j < results.n_cols; j++)
            std::cout << std::fixed << std::right
                      << "  [ " << std::setw(3) << i + 1 << " : " << std::setw(10) << labels[i].c_str()
                      << " , " << std::setw(3) << j + 1 << " : " << std::setw(10) << labels[j].c_str()
                      << " ]: " << std::setw(24) << std::setprecision(18) << results(i, j) << std::endl;

    return;

}

void print_results_with_labels(
    const arma::mat &results,
    const std::vector<std::string> &labels_1,
    const std::vector<std::string> &labels_2) {

    assert(results.n_rows == labels_1.size());
    assert(results.n_cols == labels_1.size());
    assert(results.n_rows == labels_2.size());
    assert(results.n_cols == labels_2.size());

    for (size_t i = 0; i < results.n_rows; i++)
        for (size_t j = 0; j < results.n_cols; j++)
            std::cout << std::fixed << std::right
                      << "  [ " << std::setw(3) << i + 1 << " : " << std::setw(10) << labels_1[i] << " " << labels_2[i]
                      << " , " << std::setw(3) << j + 1 << " : " << std::setw(10) << labels_1[j] << " " << labels_2[j]
                      << " ]: " << std::setw(24) << std::setprecision(18) << results(i, j) << std::endl;

    return;

}


int is_imaginary_to_b_prefactor(bool is_imaginary)
{

    if (is_imaginary)
        return -1;
    else
        return 1;

}

void join_vector(arma::vec &vc, const arma::vec &v1, const arma::vec &v2)
{

    const size_t lc = vc.n_elem;
    const size_t l1 = v1.n_elem;
    const size_t l2 = v2.n_elem;
    assert(lc == (l1 + l2));

    vc.subvec(0, l1 - 1) = v1;
    vc.subvec(l1, lc - 1) = v2;

    return;

}

void split_vector(const arma::vec &vc, arma::vec &v1, arma::vec &v2)
{

    const size_t lc = vc.n_elem;
    const size_t l1 = v1.n_elem;
    const size_t l2 = v2.n_elem;
    assert(lc == (l1 + l2));

    v1 = vc.subvec(0, l1 - 1);
    v2 = vc.subvec(l1, lc - 1);

    return;

}

bool string_to_bool(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    std::istringstream is(str);
    bool b;
    is >> std::boolalpha >> b;
    return b;
}

std::string bool_to_string(bool b) {
    return b ? "true" : "false";
}

std::string to_upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    return s;
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

void pretty_print(const arma::mat &M, std::string title, bool sci, size_t width, size_t numCols)
{
    const int extraCol = M.n_cols % numCols;
    const int fullPasses = M.n_cols / numCols;
    int precision = width - 3;

    if (title.length() != 0)
    {
        std::cout << title << std::endl;
    }

    std::ios_base::fmtflags origflags = std::cout.flags();

    if (sci)
    {
        std::cout.setf(std::ios_base::scientific, std::ios_base::floatfield);
        precision -= 4;
    }
    else
    {
        std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    }
    if (precision < 0)
        precision = 1;

    for (int i = 0; i < fullPasses; i++)
    {
        std::cout << "     ";
        for (int j = 0; j < numCols; j++)
        {
            std::cout << " " << std::setw(width-2) << (i*numCols)+1+j << "   ";
        }
        std::cout << std::endl;

        for (int j = 0; j < M.n_rows; j++)
        {
            std::cout << std::setw(5) << j+1;
            for (int k = 0; k < numCols; k++)
            {
                std::cout << "  " << std::setw(width) << std::setprecision(precision) << M(j,(i*numCols+k));
            }
            std::cout << std::endl;
        }
    }
    if (extraCol > 0)
    {
        std::cout << "     ";
        for (int j = 0; j < extraCol; j++)
        {
            std::cout << " " << std::setw(width) << (fullPasses*numCols)+j+1;
        }
        std::cout << std::endl;
        for (int j = 0; j < M.n_rows; j++)
        {
            std::cout << std::setw(5) << j+1;
            for (int k = 0; k < extraCol; k++)
            {
                std::cout << "  " << std::setw(width) << std::setprecision(precision) << M(j,(fullPasses*numCols)+k);
            }
            std::cout << std::endl;
        }
    }
    std::cout.flags(origflags);
}

void pretty_print(const arma::cube &C, std::string title, bool sci, size_t width, size_t numCols)
{
    std::stringstream slice_title;
    for (size_t s = 0; s < C.n_slices; s++) {
        slice_title << title << " [slice " << s << "]";
        pretty_print(C.slice(s), slice_title.str(), sci, width, numCols);
    }
}

// need a version check for compile-time definition (?); this is only
// for old versions of Armadillo.
namespace arma {
arma::vec regspace(int start, int delta, int end) {
    std::vector<double> v;
    double e = start;
    // this will *not* do everything that arma::regspace does!
    while (e <= end) {
        v.push_back(e);
        e += delta;
    }
    return arma::conv_to<arma::vec>::from(v);
}
}

// maybe this should just take size_t to avoid exception handling
// give me my Python dangit
arma::uvec range(int start, int stop, int step)
{

    if (start < 0 || stop < 0)
        throw std::invalid_argument("negative numbers meaningless for array indexing; no wraparound available");
    if ((stop - start) < 0)
        throw std::domain_error("no reverse ranges");
    if (step < 1)
        throw std::domain_error("???");

    return arma::conv_to<arma::uvec>::from(arma::regspace(start, step, stop - 1));

}

arma::uvec range(int start, int stop) {
    return range(start, stop, 1);
}

arma::uvec range(int stop) {
    return range(0, stop, 1);
}

template arma::uvec join<arma::uvec>(const arma::uvec &a1, const arma::uvec &a2);

double _calc_matrix_anisotropy(const arma::mat &m)
{
    if (!m.is_square())
        throw std::runtime_error("in _calc_matrix_anisotropy: matrix is not square");

    const double iso = arma::mean(arma::eig_sym(m));
    double aniso = arma::accu(m % m);
    aniso = std::sqrt(std::abs(1.5*(aniso - (3.0*iso*iso))));

    return aniso;
}

void print_polarizability(std::ostringstream &os, const arma::mat &polar_tensor)
{

    assert(polar_tensor.is_square());
    assert(polar_tensor.n_elem == 9);
    const size_t dim = 3;

    const double au_to_cubic_angstrom = pow(5.291772108e-11 * 1.0e10, 3.0);

    arma::vec principal_components;
    arma::mat orientation;
    arma::eig_sym(principal_components, orientation, polar_tensor);

    os << "  Polarizability tensor      [a.u.]" << std::endl;
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            os << " " << std::setw(14) << std::setprecision(7)
               << polar_tensor(i, j);
        }
        os << std::endl;
    }
    os << "  Principal components       [a.u.]" << std::endl;
    for (size_t i = 0; i < 3; i++) {
        os << " " << std::setw(14) << std::setprecision(7)
           << principal_components(i);
    }
    os << std::endl;
    os << "  Isotropic polarizability   [a.u.]" << std::endl;
    os << " " << std::setw(14) << std::setprecision(7)
       << (arma::trace(polar_tensor) / 3.0) << std::endl;
    os << "  Anisotropic polarizability [a.u.]" << std::endl;
    os << " " << std::setw(14) << std::setprecision(7)
       << _calc_matrix_anisotropy(polar_tensor) << std::endl;

    os << "  Polarizability tensor      [angstrom^3]" << std::endl;
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            os << " " << std::setw(14) << std::setprecision(7)
               << polar_tensor(i, j) * au_to_cubic_angstrom;
        }
        os << std::endl;
    }
    os << "  Principal components       [angstrom^3]" << std::endl;
    for (size_t i = 0; i < 3; i++) {
        os << " " << std::setw(14) << std::setprecision(7)
           << principal_components(i) * au_to_cubic_angstrom;
    }
    os << std::endl;
    os << "  Isotropic polarizability   [angstrom^3]" << std::endl;
    os << " " << std::setw(14) << std::setprecision(7)
       << (arma::trace(polar_tensor) / 3.0) * au_to_cubic_angstrom << std::endl;
    os << "  Anisotropic polarizability [angstrom^3]" << std::endl;
    os << " " << std::setw(14) << std::setprecision(7)
       << _calc_matrix_anisotropy(polar_tensor) * au_to_cubic_angstrom << std::endl;

    os << "  Orientation" << std::endl;
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            os << " " << std::setw(14) << std::setprecision(7)
               << orientation(i, j);
        }
        os << std::endl;
    }

    return;

}

void print_square_result(std::ostringstream &os, const arma::mat &square_result)
{

    assert(square_result.is_square());
    const size_t dim = square_result.n_rows;

    os << "  Result" << std::endl;
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            os << " " << std::setw(14)  << std::setprecision(7)
               << square_result(i, j);
        }
        os << std::endl;
    }
    arma::vec principal_components;
    arma::mat orientation;
    arma::eig_sym(principal_components, orientation, square_result);
    os << "  Principal components" << std::endl;
    for (size_t i = 0; i < dim; i++) {
        os << " " << std::setw(14) << std::setprecision(7)
           << principal_components(i);
    }
    os << std::endl;
    os << "  Isotropic" << std::endl;
    os << " " << std::setw(14) << std::setprecision(7)
       << (arma::trace(square_result) / static_cast<double>(dim)) << std::endl;
    os << "  Anisotropic" << std::endl;
    os << " " << std::setw(14) << std::setprecision(7)
       << _calc_matrix_anisotropy(square_result) << std::endl;
    os << "  Orientation" << std::endl;
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            os << " " << std::setw(14) << std::setprecision(7)
               << orientation(i, j);
        }
        os << std::endl;
    }

    return;

}

void printf_14_7_3by3(const arma::mat &m) {
    if (m.n_rows != 3)
        throw std::runtime_error("m.n_rows != 3");
    if (m.n_cols != 3)
        throw std::runtime_error("m.n_cols != 3");
    printf("            %14.7f %14.7f %14.7f\n",
           m(0, 0), m(0, 1), m(0, 2));
    printf("            %14.7f %14.7f %14.7f\n",
           m(1, 0), m(1, 1), m(1, 2));
    printf("            %14.7f %14.7f %14.7f\n",
           m(2, 0), m(2, 1), m(2, 2));
}

void printf_14_7_3by3_orientation(const arma::mat &m_ori) {
    if (m_ori.n_rows != 3)
        throw std::runtime_error("m_ori.n_rows != 3");
    if (m_ori.n_cols != 3)
        throw std::runtime_error("m_ori.n_cols != 3");
    printf("  Orientation:\n");
    printf("   X        %14.7f %14.7f %14.7f\n",
           m_ori(0, 0), m_ori(0, 1), m_ori(0, 2));
    printf("   Y        %14.7f %14.7f %14.7f\n",
           m_ori(1, 0), m_ori(1, 1), m_ori(1, 2));
    printf("   Z        %14.7f %14.7f %14.7f\n",
           m_ori(2, 0), m_ori(2, 1), m_ori(2, 2));
}

void printf_14_7_row(const arma::vec &v) {
    std::cout << "            ";
    for (size_t i = 0; i < v.n_elem; i++) {
        std::cout << std::fixed << std::right << std::setw(14) << std::setprecision(7) << v(i);
        if (i != (v.n_elem - 1))
            std::cout << " ";
    }
    std::cout << std::endl;
}
