#include <cassert>
#include <ostream>
#include <stdexcept>
#include <sstream>

#include "constants.h"
#include "linear/helpers.h"
#include "operator_spec.h"

namespace libresponse {

void operator_spec::init_indices(
    const arma::umat &fragment_occupations,
    const libresponse::configurable &cfg) {

    const arma::uvec nbasis_frgm = fragment_occupations.col(0);
    const arma::uvec norb_frgm = fragment_occupations.col(1);
    const arma::uvec nocc_frgm_alph = fragment_occupations.col(2);
    const arma::uvec nocc_frgm_beta = fragment_occupations.col(3);
    const arma::uvec nvirt_frgm_alph = norb_frgm - nocc_frgm_alph;
    const arma::uvec nvirt_frgm_beta = norb_frgm - nocc_frgm_beta;

    indices_ao = make_indices_ao(nbasis_frgm);
    const int frgm_response_idx = cfg.get_param<int>("_frgm_response_idx");
    if (frgm_response_idx > 0) {
        const type::indices indices_mo_allfrgm_alph = make_indices_mo_restricted_local_occ_all_virt(nocc_frgm_alph, nvirt_frgm_alph);
        const type::indices indices_mo_allfrgm_beta = make_indices_mo_restricted_local_occ_all_virt(nocc_frgm_beta, nvirt_frgm_beta);
        indices_mo_alph = indices_mo_allfrgm_alph.at(frgm_response_idx - 1);
        indices_mo_beta = indices_mo_allfrgm_beta.at(frgm_response_idx - 1);
    } else {
        indices_mo_alph = make_indices_mo_restricted(nocc_frgm_alph, nvirt_frgm_alph);
        indices_mo_beta = make_indices_mo_restricted(nocc_frgm_beta, nvirt_frgm_beta);
    }
}

std::ostream& operator<<(std::ostream &stream, const operator_metadata &om) {
    stream << " operator_label: " << om.operator_label << " origin_label: " << om.origin_label << " slice_idx: " << om.slice_idx << " is_imaginary: " << om.is_imaginary << " is_spin_dependent: " << om.is_spin_dependent;
    return stream;
}

void operator_spec::form_rhs(
    const arma::cube &C,
    const arma::uvec &occupations,
    const libresponse::configurable &cfg) {

    if (cfg.has_param("prefix"))
        prefix = cfg.get_param("prefix");
    else
        prefix = "";

    const bool mask_operator_ao = cfg.get_param<bool>("_mask_operator_ao");
    const bool mask_rhsvec_mo = cfg.get_param<bool>("_mask_rhsvec_mo");

    const size_t norb = C.n_cols;
    const size_t nden = C.n_slices;
    has_beta = (nden == 2);
    const size_t nocc_alph = occupations(0);
    const size_t nvirt_alph = occupations(1);
    const size_t nocc_beta = occupations(2);
    const size_t nvirt_beta = occupations(3);
    arma::mat C_occ_alph = C.slice(0).cols(0, nocc_alph - 1);
    arma::mat C_virt_alph = C.slice(0).cols(nocc_alph, norb - 1);
    arma::mat C_occ_beta;
    arma::mat C_virt_beta;
    if (nden == 2) {
        C_occ_beta = C.slice(1).cols(0, nocc_beta - 1);
        C_virt_beta = C.slice(1).cols(nocc_beta, norb - 1);
    }
    const size_t nov_alph = nocc_alph * nvirt_alph;
    const size_t nov_beta = nocc_beta * nvirt_beta;

    integrals_mo_ai_alph.set_size(nov_alph, integrals_ao.n_slices);
    if (mask_operator_ao)
        libresponse::one_electron_mn_mats_to_ia_vecs(integrals_mo_ai_alph, integrals_ao, C_occ_alph, C_virt_alph, indices_ao);
    else
        libresponse::one_electron_mn_mats_to_ia_vecs(integrals_mo_ai_alph, integrals_ao, C_occ_alph, C_virt_alph);
    if (nden == 2) {
        integrals_mo_ai_beta.set_size(nov_beta, integrals_ao.n_slices);
        if (mask_operator_ao)
            libresponse::one_electron_mn_mats_to_ia_vecs(integrals_mo_ai_beta, integrals_ao, C_occ_beta, C_virt_beta, indices_ao);
        else
            libresponse::one_electron_mn_mats_to_ia_vecs(integrals_mo_ai_beta, integrals_ao, C_occ_beta, C_virt_beta);
    }

    if (mask_rhsvec_mo) {
        arma::mat rhsvec_masked_alph(integrals_mo_ai_alph.n_rows, integrals_mo_ai_alph.n_cols, arma::fill::zeros);
        const arma::uvec indices_allcols = range(integrals_mo_ai_alph.n_cols);
        rhsvec_masked_alph(indices_mo_alph, indices_allcols) = integrals_mo_ai_alph(indices_mo_alph, indices_allcols);
        integrals_mo_ai_alph = rhsvec_masked_alph;
        if (nden == 2) {
            arma::mat rhsvec_masked_beta(integrals_mo_ai_beta.n_rows, integrals_mo_ai_beta.n_cols, arma::fill::zeros);
            rhsvec_masked_beta(indices_mo_beta, indices_allcols) = integrals_mo_ai_beta(indices_mo_beta, indices_allcols);
            integrals_mo_ai_beta = rhsvec_masked_beta;
        }
    }

    // -V on RHS. 1 is for singly-occupied orbitals, 2 for
    // -doubly-occupied. For singly-occupied orbitals, a final
    // -multiplication by 2 is necessary at the very end.
    if (nden == 2) {
        integrals_mo_ai_alph *= -1.0;
        integrals_mo_ai_beta *= -1.0;
    } else {
        integrals_mo_ai_alph *= -2.0;
    }

    // Scale spin-orbit integrals like DALTON.
    const double hsofac = std::pow(libresponse::constant::alpha, 2.0) / 4.0;
    const size_t found = metadata.operator_label.find("spinorb");
    if (found != std::string::npos) {
        integrals_mo_ai_alph *= hsofac;
        if (nden == 2)
            integrals_mo_ai_beta *= hsofac;
    }

    // Might as well allocate space for the response vectors at this
    // point since we already know whether or not response will be
    // calculated.
    if (do_response) {
        rspvecs_alph.set_size(nov_alph, integrals_ao.n_slices);
        if (nden == 2)
            rspvecs_beta.set_size(nov_beta, integrals_ao.n_slices);
    }
}

void operator_spec::form_guess_rspvec(const arma::vec &ediff, double frequency, bool beta) {
    // The initial guess for the response vectors is the uncoupled
    // result. If response vectors were read in from disk, then they
    // serve as the guess, which should not be formed. The guess
    // should also not be formed if not doing response.
    if (do_response) {
        const size_t len = ediff.n_elem;
        double * rspvec_ptr = NULL;
        double * rhsvec_ptr = NULL;
        for (size_t s = 0; s < integrals_ao.n_slices; s++) {
            if (!beta) {
                rspvec_ptr = rspvecs_alph.colptr(s);
                rhsvec_ptr = integrals_mo_ai_alph.colptr(s);
            } else {
                rspvec_ptr = rspvecs_beta.colptr(s);
                rhsvec_ptr = integrals_mo_ai_beta.colptr(s);
            }
            arma::vec rspvec(rspvec_ptr, len, false, true);
            const arma::vec rhsvec(rhsvec_ptr, len, false, true);
            libresponse::form_guess_rspvec(rspvec, rhsvec, ediff, frequency);
        }
    }
}

void operator_spec::form_guess_rspvec(const arma::mat &ediff, double frequency, bool beta, size_t nov, const libresponse::configurable &cfg) {
    // The initial guess for the response vectors is the uncoupled
    // result. If response vectors were read in from disk, then they
    // serve as the guess, which should not be formed. The guess
    // should also not be formed if not doing response.
    if (do_response) {
        const bool mask_rspvec_guess_mo = cfg.get_param<bool>("_mask_rspvec_guess_mo");
        assert(ediff.n_rows == ediff.n_cols);
        const size_t len = ediff.n_rows;
        double * rspvec_full_ptr = NULL;
        double * rspvec_ptr = NULL;
        double * rhsvec_ptr = NULL;
        arma::uvec indices_mo;
        if (!beta)
            indices_mo = indices_mo_alph;
        else
            indices_mo = indices_mo_beta;
        const bool reduce = cfg.get_param<bool>("_mask_ediff_mo");
        const size_t nred = indices_mo.n_elem;
        arma::vec rspvec_reduced, rhsvec_reduced;
        arma::uvec indices_mo_red;
        if (reduce) {
            assert(len == nred);
            indices_mo_red = range(nred);
            rspvec_reduced.set_size(nred);
            rhsvec_reduced.set_size(nred);
        } else {
            assert(len == nov);
        }
        arma::uvec vs(1);
        for (size_t s = 0; s < integrals_ao.n_slices; s++) {
            vs(0) = s;
            if (!beta) {
                if (reduce) {
                    rspvec_reduced = rspvecs_alph.submat(indices_mo, vs);
                    rhsvec_reduced = integrals_mo_ai_alph.submat(indices_mo, vs);
                    rspvec_full_ptr = rspvecs_alph.colptr(s);
                    rspvec_ptr = rspvec_reduced.memptr();
                    rhsvec_ptr = rhsvec_reduced.memptr();
                } else {
                    rspvec_ptr = rspvecs_alph.colptr(s);
                    rhsvec_ptr = integrals_mo_ai_alph.colptr(s);
                }
            } else {
                if (reduce) {
                    rspvec_reduced = rspvecs_beta.submat(indices_mo, vs);
                    rhsvec_reduced = integrals_mo_ai_beta.submat(indices_mo, vs);
                    rspvec_full_ptr = rspvecs_beta.colptr(s);
                    rspvec_ptr = rspvec_reduced.memptr();
                    rhsvec_ptr = rhsvec_reduced.memptr();
                } else {
                    rspvec_ptr = rspvecs_beta.colptr(s);
                    rhsvec_ptr = integrals_mo_ai_beta.colptr(s);
                }
            }
            arma::vec rspvec(rspvec_ptr, len, false, true);
            const arma::vec rhsvec(rhsvec_ptr, len, false, true);
            libresponse::form_guess_rspvec(rspvec, rhsvec, ediff, frequency);

            if (reduce) {
                assert(rspvec_full_ptr != NULL);
                arma::vec rspvec_full(rspvec_full_ptr, nov);
                rspvec_full.zeros();
                rspvec_full(indices_mo) = rspvec(indices_mo_red);
            } else if (mask_rspvec_guess_mo) {
                arma::vec rspvec_masked(rspvec.n_elem, arma::fill::zeros);
                rspvec_masked(indices_mo) = rspvec(indices_mo);
                rspvec = rspvec_masked;
            }
        }
    }
}

void operator_spec::save_to_disk(int save_level, bool is_guess) {
    std::stringstream ss_rhs_alph;
    std::stringstream ss_rsp_alph;
    std::stringstream ss_rhs_beta;
    std::stringstream ss_rsp_beta;
    if (save_level >= 1) {
        ss_rsp_alph.str(std::string());
        ss_rhs_alph.str(std::string());
        ss_rhs_alph << prefix << "rhsvecs_" << metadata.operator_label << "_mo_alph.dat";
        if (is_guess)
            ss_rsp_alph << prefix << "rspvecs_guess_" << metadata.operator_label << "_mo_alph.dat";
        else
            ss_rsp_alph << prefix << "rspvecs_" << metadata.operator_label << "_mo_alph.dat";
        integrals_mo_ai_alph.save(ss_rhs_alph.str(), arma::arma_ascii);
        rspvecs_alph.save(ss_rsp_alph.str(), arma::arma_ascii);
        if (has_beta) {
            ss_rhs_beta.str(std::string());
            ss_rsp_beta.str(std::string());
            ss_rhs_beta << prefix << "rhsvecs_" << metadata.operator_label << "_mo_beta.dat";
            if (is_guess)
                ss_rsp_beta << prefix << "rspvecs_guess_" << metadata.operator_label << "_mo_beta.dat";
            else
                ss_rsp_beta << prefix << "rspvecs_" << metadata.operator_label << "_mo_beta.dat";
            rspvecs_beta.save(ss_rsp_beta.str(), arma::arma_ascii);
            integrals_mo_ai_beta.save(ss_rhs_beta.str(), arma::arma_ascii);
        }
        // save = 2 -> also write out in AO basis.
        // TODO rhsvecs in AO basis, why is this commented out?
        // if (save_level >= 2) {
        //     ss_rhs_alph.str(std::string());
        //     ss_rsp_alph.str(std::string());
        //     if (is_guess)
        //         ss_rsp_alph << prefix << "rspvecs_guess_" << metadata.operator_label << "_ao_alph.dat";
        //     else
        //         ss_rsp_alph << prefix << "rspvecs_" << metadata.operator_label << "_ao_alph.dat";
        //     const size_t ncomponents = rspvecs_alph.n_cols;
        //     arma::cube rspvecs_ao_alph(nbasis, nbasis, ncomponents);
        //     libresponse::one_electron_ia_vecs_to_mn_mats(rspvecs_ao_alph, rspvecs_alph, C_occ_alph, C_virt_alph);
        //     rspvecs_ao_alph.save(ss_rsp_alph.str(), arma::arma_ascii);
        //     if (has_beta) {
        //         ss_rhs_beta.str(std::string());
        //         ss_rsp_beta.str(std::string());
        //         if (is_guess)
        //             ss_rsp_beta << prefix << "rspvecs_guess_" << metadata.operator_label << "_ao_beta.dat";
        //         else
        //             ss_rsp_beta << prefix << "rspvecs_" << metadata.operator_label << "_ao_beta.dat";
        //         arma::cube rspvecs_ao_beta(nbasis, nbasis, ncomponents);
        //         libresponse::one_electron_ia_vecs_to_mn_mats(rspvecs_ao_beta, rspvecs_beta, C_occ_beta, C_virt_beta);
        //         rspvecs_ao_beta.save(ss_rsp_beta.str(), arma::arma_ascii);
        //     }
        // }
    }
}

std::vector<std::string> make_operator_label_vec(const std::vector<operator_spec> &operators)
{

    if (operators.empty())
        throw std::runtime_error("operators.empty()");

    std::vector<std::string> labels;

    for (size_t i = 0; i < operators.size(); i++)
        for (size_t s = 0; s < operators[i].integrals_ao.n_slices; s++)
            labels.push_back(operators[i].metadata.operator_label);

    return labels;

}

std::vector<std::string> make_operator_component_vec(const std::vector<operator_spec> &operators)
{

    if (operators.empty())
        throw std::runtime_error("operators.empty()");

    std::vector<std::string> labels;

    for (size_t i = 0; i < operators.size(); i++)
        for (size_t s = 0; s < operators[i].integrals_ao.n_slices; s++) {
            std::stringstream ss;
            ss << (s + 1);
            labels.push_back(ss.str());
        }

    return labels;

}

} // namespace libresponse
