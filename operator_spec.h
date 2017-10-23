#ifndef LIBRESPONSE_OPERATOR_SPEC_H_
#define LIBRESPONSE_OPERATOR_SPEC_H_

/*!
 * @file
 *
 * Hold operators (integrals and metadata) and lists of operators.
 */

#include "configurable.h"
#include "indices.h"
#include "utils.h"

namespace libresponse {

/*!
 * Internal representation of operator/integral, metadata only.
 *
 * The operator and origin labels are used for printing to the screen,
 * not for actual computation.
 *
 * The slice ID is used to indicate if only a single slice out of the
 * integrals is to be used. If this is -1, all slices (components)
 * will be used.
 *
 * @todo Is slice_idx/slice_idx_ even used anywhere?
 */
class operator_metadata {

public:
    //! name of the operator
    std::string operator_label;
    //! plain-text description of the operator origin
    std::string origin_label;
    //! index (zero-based) of cube slice to keep integrals from
    int slice_idx;
    //! Are the integrals imaginary (stored antisymmetric)?
    bool is_imaginary;
    //! Do the integrals have a spin factor in them (alpha/beta will
    //! be opposite sign in MO basis)?
    bool is_spin_dependent;

public:
    /*!
     * Default constructor.
     */
    operator_metadata(
        std::string &operator_label_,
        std::string &origin_label_,
        int slice_idx_,
        bool is_imaginary_,
        bool is_spin_dependent_)
        : operator_label(operator_label_)
        , origin_label(origin_label_)
        , slice_idx(slice_idx_)
        , is_imaginary(is_imaginary_)
        , is_spin_dependent(is_spin_dependent_)
        { }

};

std::ostream& operator<<(std::ostream &stream, const operator_metadata &om);

/*!
 * Information about an operator (integrals and associated metadata).
 *
 * The one-electron integrals are stored as [nbsf, nbsf] matrices
 * in an Armadillo cube, where each slice is a part of the
 * operator. If necessary, the ordering is slices per atom, then
 * Cartesian components x, y, z, xx, xy, xz, yx, yy, ...
 *
 * The B prefactor is only needed for RPA (not TDA/CIS). If the
 * operator is real, form \f$(\mathbf{A} + \mathbf{B})\f$. If the
 * operator is imaginary, form \f$(\mathbf{A} - \mathbf{B})\f$.
 *
 * @todo A single operator might have more than one origin, in the
 * case of nuclear centers. Is it important to store those, such
 * as a std::vector<arma::vec>?
 */
class operator_spec {

public:

    operator_metadata metadata; //!< operator metadata
    arma::cube integrals_ao;    //!< AO-basis integrals for operator
    arma::vec origin;           //!< operator (integral) origin

    //! Should this operator be used as a property gradient on the RHS
    //! of the response equations?
    bool do_response;

    // TODO sign is flipped?
    int b_prefactor; //!< prefactor for RPA B-matrix; 1/-1 for real/imaginary

    std::string prefix;

    bool has_beta;
    arma::mat integrals_mo_ai_alph;
    arma::mat integrals_mo_ai_beta;
    void form_rhs(
        const arma::cube &C,
        const arma::uvec &occupations,
        const libresponse::configurable &cfg);
    arma::mat rspvecs_alph;
    arma::mat rspvecs_beta;
    void form_guess_rspvec(const arma::vec &ediff, double frequency, bool beta);
    type::indices indices_ao;
    arma::uvec indices_mo_alph;
    arma::uvec indices_mo_beta;
    void form_guess_rspvec(const arma::mat &ediff, double frequency, bool beta, size_t nov, const libresponse::configurable &cfg);
    void save_to_disk(int save_level, bool is_guess);

public:

    /*!
     * Default constructor.
     *
     * @param[in] &operator_metadata_ information about the operator
     * @param[in] &integrals_ao_ AO-basis integrals for operator
     * @param[in] &origin_ operator (integral) origin
     */
    operator_spec(
        const operator_metadata &metadata_,
        arma::cube &integrals_ao_,
        arma::vec &origin_,
        bool do_response_)
        : metadata(metadata_)
        , integrals_ao(integrals_ao_)
        , origin(origin_)
        , do_response(do_response_)
        {
            b_prefactor = is_imaginary_to_b_prefactor(metadata.is_imaginary);

            // Modify the metadata so the origin label also contains
            // the numerical operator origin (in bohr).
            std::ostringstream origin_label;
            origin_label << metadata.origin_label << \
                " (" << origin(0) << \
                ", " << origin(1) << \
                ", " << origin(2) << ")";
            metadata.origin_label = origin_label.str();
        }

    void init_indices(
        const arma::umat &fragment_occupations,
        const libresponse::configurable &cfg);

};

/*!
 * A shorter way to make a list of labels from a list of operators.
 *
 * @param[in] &operators operators to grab labels from
 *
 * @returns vector of labels taken from each operator
 */
std::vector<std::string> make_operator_label_vec(const std::vector<operator_spec> &operators);

/*!
 *
 *
 * @todo document me!
 */
std::vector<std::string> make_operator_component_vec(const std::vector<operator_spec> &operators);

} // namespace libresponse

#endif // LIBRESPONSE_OPERATOR_SPEC_H_
