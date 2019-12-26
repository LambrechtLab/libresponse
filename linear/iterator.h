#ifndef LIBRESPONSE_LINEAR_ITERATOR_H_
#define LIBRESPONSE_LINEAR_ITERATOR_H_

/*!
 * @file
 *
 * ...
 */

#include "helpers.h"
#include "printing.h"
#include "../matvec_i.h"

namespace libresponse {

template <class T>
class SolverIterator_i {

protected:

    // Intermediates that aren't needed outside the iterations:
    // Dg, J, K, products, ints_ovov, ints_mnov, rspvecs_old

    // Hold the generalized density (the response vector in the AO
    // basis) and the resulting J/K formed from that density.
    arma::cube Dg;
    arma::cube J;
    arma::cube K;

    // Do compute the generalized density Dg, or keep L and R
    // separately?
    bool do_compute_generalized_density;
    // TODO comment me!
    std::vector<arma::mat> L;
    std::vector<arma::mat> R;
    arma::mat L_alph;
    arma::mat L_beta;
    arma::mat R_alph;
    arma::mat R_beta;

    // These are for the intermediate integrals (2 transformed
    // indices).
    arma::cube ints_mnov;
    // These are for the 4-index transformed integrals; upon
    // repacking into a vector, these are the matrix-vector
    // products (see above).
    arma::mat ints_ovov_alph;
    arma::mat ints_ovov_beta;

    size_t nden;
    // These are present in the initialization.
    libresponse::configurable * cfg;
    std::vector<operator_spec> * operators;
    MatVec_i * matvec;
    arma::cube * C;
    T * ediff_alph;
    T * ediff_beta;
    double frequency;
    int maxiter;
    double conv;

    size_t nocc_alph, nvirt_alph, nocc_beta, nvirt_beta;
    size_t nov_alph, nov_beta;

    int print_level;

public:

    SolverIterator_i() { }
    virtual ~SolverIterator_i() { }

    void set_orbital_occupations(
        size_t nocc_alph_, size_t nvirt_alph_,
        size_t nocc_beta_, size_t nvirt_beta_
        )
        {

            nocc_alph = nocc_alph_;
            nvirt_alph = nvirt_alph_;
            nocc_beta = nocc_beta_;
            nvirt_beta = nvirt_beta_;

            nov_alph = nocc_alph * nvirt_alph;
            nov_beta = nocc_beta * nvirt_beta;

            return;

        }

    void init(
        std::vector<operator_spec> * operators_,
        configurable * cfg_,
        MatVec_i * matvec_,
        arma::cube * C_,
        T * ediff_alph_,
        T * ediff_beta_,
        double frequency_,
        int maxiter_,
        double conv_
        )
        {

            operators = operators_;
            cfg = cfg_;
            matvec = matvec_;
            C = C_;
            ediff_alph = ediff_alph_;
            ediff_beta = ediff_beta_;
            frequency = frequency_;
            maxiter = maxiter_;
            conv = conv_;

            nden = C->n_slices;

            const size_t nbasis = C->n_rows;
            do_compute_generalized_density = cfg->get_param<bool>("_do_compute_generalized_density");
            if (do_compute_generalized_density)
                Dg.set_size(nbasis, nbasis, nden);
            else {
                L_alph.set_size(nbasis, nvirt_alph);
                R_alph.set_size(nbasis, nvirt_alph);
                L.push_back(L_alph);
                R.push_back(R_alph);
                if (nden == 2) {
                    L_beta.set_size(nbasis, nvirt_beta);
                    R_beta.set_size(nbasis, nvirt_beta);
                    L.push_back(L_beta);
                    R.push_back(R_beta);
                }
            }
            J.set_size(nbasis, nbasis, nden);
            K.set_size(nbasis, nbasis, nden);
            ints_mnov.set_size(nbasis, nbasis, nden);
            ints_ovov_alph.set_size(nvirt_alph, nocc_alph);
            if (nden == 2) {
                ints_ovov_beta.set_size(nvirt_beta, nocc_beta);
            }

            print_level = std::atoi(cfg->get_param("print_level").c_str());

        }


    virtual void run() { }

};

class SolverIterator_linear : public SolverIterator_i<arma::vec> {

public:

    void run() {

        // These are for the previous iteration's response vectors,
        // used for calculating the RMSD between steps.
        std::vector< arma::mat > rspvecs_old_alph;
        for (size_t i = 0; i < operators->size(); i++) {
            if (operators->at(i).do_response)
                rspvecs_old_alph.push_back(arma::mat(nov_alph, operators->at(i).integrals_ao.n_slices));
        }
        std::vector< arma::mat > rspvecs_old_beta;
        // These are for the matrix-vector products between the
        // orbital Hessian and the trial/guess response vector at a
        // single iteration.
        std::vector< arma::mat > products_alph;
        for (size_t i = 0; i < operators->size(); i++) {
            if (operators->at(i).do_response)
                products_alph.push_back(arma::mat(nov_alph, operators->at(i).integrals_ao.n_slices, arma::fill::zeros));
        }
        std::vector< arma::mat > products_beta;
        if (nden == 2) {
            for (size_t i = 0; i < operators->size(); i++) {
                if (operators->at(i).do_response) {
                    rspvecs_old_beta.push_back(arma::mat(nov_beta, operators->at(i).integrals_ao.n_slices));
                    products_beta.push_back(arma::mat(nov_beta, operators->at(i).integrals_ao.n_slices, arma::fill::zeros));
                }
            }
        }

        iteration_info_linear info;
        if (nden == 1)
            info.has_beta = false;
        else
            info.has_beta = true;

        const std::string hamiltonian = cfg->get_param("hamiltonian");
        const std::string spin = cfg->get_param("spin");
        arma::mat C_occ_alph = C->slice(0).cols(0, nocc_alph - 1);
        arma::mat C_virt_alph = C->slice(0).cols(nocc_alph, nocc_alph + nvirt_alph - 1);
        arma::mat C_occ_beta;
        arma::mat C_virt_beta;
        if (nden == 2) {
            C_occ_beta = C->slice(1).cols(0, nocc_beta - 1);
            C_virt_beta = C->slice(1).cols(nocc_beta, nocc_beta + nvirt_beta - 1);
        }

        // Perform the linear CPSCF iterations. Loop over operators,
        // then components of that operator, converging each one
        // separately.
        for (size_t i = 0; i < operators->size(); i++) {

            if (operators->at(i).do_response) {

                const int b_prefactor = operators->at(i).b_prefactor;

                for (size_t s = 0; s < operators->at(i).integrals_ao.n_slices; s++) {

                    bool is_converged = false;

                    if (print_level >= 2) {
                        std::cout << \
                            "  Operator: " <<                            \
                            operators->at(i).metadata.operator_label <<  \
                            " / component: " << s + 1 <<                 \
                            " / origin: " <<                             \
                            operators->at(i).metadata.origin_label <<    \
                            std::endl;
                    }

                    // Wrappers over vectors.
                    arma::vec product_alph(products_alph[i].colptr(s), nov_alph, false, true);
                    arma::vec rspvec_alph(operators->at(i).rspvecs_alph.colptr(s), nov_alph, false, true);
                    arma::vec rspvec_old_alph(rspvecs_old_alph[i].colptr(s), nov_alph, false, true);
                    const arma::vec rhsvec_alph(operators->at(i).integrals_mo_ai_alph.colptr(s), nov_alph, false, true);

                    // The Armadillo vectors need to pass through to
                    // the correct memory locations, but the pointers
                    // are undefined for 1 density since the
                    // std::vectors are unpopulated.
                    double * product_beta_ptr = NULL;
                    double * rspvec_beta_ptr = NULL;
                    double * rspvec_old_beta_ptr = NULL;
                    double * rhsvec_beta_ptr = NULL;
                    if (nden == 2) {
                        product_beta_ptr = products_beta[i].colptr(s);
                        rspvec_beta_ptr = operators->at(i).rspvecs_beta.colptr(s);
                        rspvec_old_beta_ptr = rspvecs_old_beta[i].colptr(s);
                        rhsvec_beta_ptr = operators->at(i).integrals_mo_ai_beta.colptr(s);
                    }
                    // This is safe even with null pointers, because I'm a
                    // careful programmer...sometimes?
                    arma::vec product_beta(product_beta_ptr, nov_beta, false, true);
                    arma::vec rspvec_beta(rspvec_beta_ptr, nov_beta, false, true);
                    arma::vec rspvec_old_beta(rspvec_old_beta_ptr, nov_beta, false, true);
                    const arma::vec rhsvec_beta(rhsvec_beta_ptr, nov_beta, false, true);

                    // Perform the response vector update.
                    info.max_rmsd_alph = 0.0;
                    info.max_rmsd_beta = 0.0;
                    for (size_t iter = 0; iter < maxiter; iter++) {

                        if (print_level >= 10) {
                            rspvec_alph.print("rspvec_old_alph");
                            if (nden == 2)
                                rspvec_beta.print("rspvec_old_beta");
                        }

                        if (do_compute_generalized_density) {
                            // Compute J and K from D.
                            compute_generalized_density(Dg.slice(0), rspvec_alph, C_occ_alph, C_virt_alph);
                            if (nden == 2)
                                compute_generalized_density(Dg.slice(1), rspvec_beta, C_occ_beta, C_virt_beta);

                            if (print_level >= 10)
                                pretty_print(Dg, "Dg");

                            matvec->compute(J, K, Dg);
                        } else {
                            // Compute J and K from L and R.
                            L[0] = C_virt_alph;
                            arma::mat qm_alph(rspvec_alph.memptr(), nvirt_alph, nocc_alph, false, true);
                            R[0] = (qm_alph * C_occ_alph.t()).t();
                            if (nden == 2) {
                                L[1] = C_virt_beta;
                                arma::mat qm_beta(rspvec_beta.memptr(), nvirt_beta, nocc_beta, false, true);
                                R[1] = (qm_beta * C_occ_beta.t()).t();
                            }
                            matvec->compute(J, K, L, R);
                        }

                        if (print_level >= 10) {
                            pretty_print(J, "J");
                            pretty_print(K, "K");
                        }

                        form_orbital_hessian_equations(ints_mnov, J, K, hamiltonian, spin, b_prefactor);

                        if (print_level >= 10)
                            pretty_print(ints_mnov, "ints_mnov");

                        AO2MO(ints_ovov_alph, ints_mnov.slice(0), C_virt_alph, C_occ_alph);
                        if (nden == 2)
                            AO2MO(ints_ovov_beta, ints_mnov.slice(1), C_virt_beta, C_occ_beta);

                        repack_matrix_to_vector(product_alph, ints_ovov_alph);
                        if (nden == 2)
                            repack_matrix_to_vector(product_beta, ints_ovov_beta);

                        if (print_level >= 10) {
                            product_alph.print("product_alph");
                            if (nden == 2)
                                product_beta.print("product_beta");
                        }

                        form_new_rspvec(rspvec_alph, product_alph, rhsvec_alph, *ediff_alph, frequency);
                        if (nden == 2)
                            form_new_rspvec(rspvec_beta, product_beta, rhsvec_beta, *ediff_beta, frequency);

                        if (print_level >= 10) {
                            rspvec_alph.print("rspvec_alph");
                            if (nden == 2)
                                rspvec_beta.print("rspvec_beta");
                        }

                        // Compute and check for convergence.
                        info.curr_rmsd_alph = rmsd(rspvec_alph, rspvec_old_alph);
                        if (nden == 2) {
                            info.curr_rmsd_beta = rmsd(rspvec_beta, rspvec_old_beta);
                        }
                        info.iter = iter + 1;
                        info.s = s + 1;
                        if (print_level >= 2) {
                            std::cout << info << std::endl;
                            // arma::vec diff = rspvec_alph - rspvec_old_alph;
                            // std::cout << " inf:" << arma::norm(diff, "inf") << std::endl;
                            // std::cout << "frob:" << arma::norm(diff, "frob") << std::endl;
                            // std::cout << "   2:" << arma::norm(diff, 2) << std::endl;
                            // if (nden == 2) {
                            //     diff = rspvec_alph - rspvec_old_alph;
                            //     std::cout << " inf:" << arma::norm(diff, "inf") << std::endl;
                            //     std::cout << "frob:" << arma::norm(diff, "frob") << std::endl;
                            //     std::cout << "   2:" << arma::norm(diff, 2) << std::endl;
                            // }
                        }
                        if (info.curr_rmsd_alph < conv) {
                            if (nden == 1) {
                                is_converged = true;
                                break;
                            } else if (info.curr_rmsd_beta < conv) {
                                is_converged = true;
                                break;
                            }
                        }

                        rspvec_old_alph = rspvec_alph;
                        if (nden == 2)
                            rspvec_old_beta = rspvec_beta;

                    }

                    // If not converged after the maximum number of
                    // iterations, crash.
                    if (!is_converged) {
                        throw std::runtime_error("not converged after max iterations");
                    }

                }

            }

        }

        return;

    }

};

class SolverIterator_nonorthogonal : public SolverIterator_i<arma::mat> {

protected:

    arma::umat fragment_occupations;
    size_t nbasis;

public:

    void set_fragment_occupations(
        const arma::umat &fragment_occupations_
        ) {
        // if (arma::accu(fragment_occupations_.col(0)) != Dg.n_rows)
        //     throw 1;
        fragment_occupations = fragment_occupations_;
        // Used anywhere?
        nbasis = arma::accu(fragment_occupations.col(0));

        return;
    }

};

class SolverIterator_ALMO_linear : public SolverIterator_nonorthogonal {

public:

    void run() {

        arma::cube Dg_masked;
        arma::cube J_masked;
        arma::cube K_masked;
        arma::cube ints_mnov_masked;

        const arma::uvec nbasis_frgm = fragment_occupations.col(0);
        const arma::uvec norb_frgm = fragment_occupations.col(1);
        const arma::uvec nocc_frgm_alph = fragment_occupations.col(2);
        const arma::uvec nocc_frgm_beta = fragment_occupations.col(3);
        const arma::uvec nvirt_frgm_alph = norb_frgm - nocc_frgm_alph;
        const arma::uvec nvirt_frgm_beta = norb_frgm - nocc_frgm_beta;

        const type::indices indices_ao = make_indices_ao(nbasis_frgm);
        arma::uvec indices_mo_alph, indices_mo_beta;
        const int frgm_response_idx = cfg->get_param<int>("_frgm_response_idx");
        if (frgm_response_idx > 0) {
            const type::indices indices_mo_allfrgm_alph = make_indices_mo_restricted_local_occ_all_virt(nocc_frgm_alph, nvirt_frgm_alph);
            const type::indices indices_mo_allfrgm_beta = make_indices_mo_restricted_local_occ_all_virt(nocc_frgm_beta, nvirt_frgm_beta);
            indices_mo_alph = indices_mo_allfrgm_alph.at(frgm_response_idx - 1);
            indices_mo_beta = indices_mo_allfrgm_beta.at(frgm_response_idx - 1);
        } else {
            indices_mo_alph = make_indices_mo_restricted(nocc_frgm_alph, nvirt_frgm_alph);
            indices_mo_beta = make_indices_mo_restricted(nocc_frgm_beta, nvirt_frgm_beta);
        }

        arma::uvec indices_mo_red_alph, indices_mo_red_beta;
        size_t nred_alph, nred_beta;
        arma::vec rhsvec_reduced_alph, rhsvec_reduced_beta;
        arma::vec product_reduced_alph, product_reduced_beta;
        arma::vec rspvec_reduced_alph, rspvec_reduced_beta;

        const bool reduce = cfg->get_param<bool>("_mask_ediff_mo");
        if (reduce) {
            nred_alph = indices_mo_alph.n_elem;
            indices_mo_red_alph = range(nred_alph);
            rhsvec_reduced_alph.set_size(nred_alph);
            product_reduced_alph.set_size(nred_alph);
            rspvec_reduced_alph.set_size(nred_alph);
            if (nden == 2) {
                nred_beta = indices_mo_beta.n_elem;
                indices_mo_red_beta = range(nred_beta);
                rhsvec_reduced_beta.set_size(nred_beta);
                product_reduced_beta.set_size(nred_beta);
                rspvec_reduced_beta.set_size(nred_beta);
            }
        }

        // These are for the previous iteration's response vectors,
        // used for calculating the RMSD between steps.
        std::vector< arma::mat > rspvecs_old_alph;
        for (size_t i = 0; i < operators->size(); i++) {
            if (operators->at(i).do_response)
                rspvecs_old_alph.push_back(arma::mat(nov_alph, operators->at(i).integrals_ao.n_slices));
        }
        std::vector< arma::mat > rspvecs_old_beta;
        // These are for the matrix-vector products between the
        // orbital Hessian and the trial/guess response vector at a
        // single iteration.
        std::vector< arma::mat > products_alph;
        for (size_t i = 0; i < operators->size(); i++) {
            if (operators->at(i).do_response)
                products_alph.push_back(arma::mat(nov_alph, operators->at(i).integrals_ao.n_slices, arma::fill::zeros));
        }
        std::vector< arma::mat > products_beta;
        if (nden == 2) {
            for (size_t i = 0; i < operators->size(); i++) {
                if (operators->at(i).do_response) {
                    rspvecs_old_beta.push_back(arma::mat(nov_beta, operators->at(i).integrals_ao.n_slices));
                    products_beta.push_back(arma::mat(nov_beta, operators->at(i).integrals_ao.n_slices, arma::fill::zeros));
                }
            }
        }

        iteration_info_linear info;
        if (nden == 1)
            info.has_beta = false;
        else
            info.has_beta = true;

        const std::string hamiltonian = cfg->get_param("hamiltonian");
        const std::string spin = cfg->get_param("spin");
        arma::mat C_occ_alph = C->slice(0).cols(0, nocc_alph - 1);
        arma::mat C_virt_alph = C->slice(0).cols(nocc_alph, nocc_alph + nvirt_alph - 1);
        arma::mat C_occ_beta;
        arma::mat C_virt_beta;
        if (nden == 2) {
            C_occ_beta = C->slice(1).cols(0, nocc_beta - 1);
            C_virt_beta = C->slice(1).cols(nocc_beta, nocc_beta + nvirt_beta - 1);
        }

        // Perform the linear CPSCF iterations. Loop over operators,
        // then components of that operator, converging each one
        // separately.
        for (size_t i = 0; i < operators->size(); i++) {

            if (operators->at(i).do_response) {

                const int b_prefactor = operators->at(i).b_prefactor;

                for (size_t s = 0; s < operators->at(i).integrals_ao.n_slices; s++) {

                    bool is_converged = false;

                    if (print_level >= 2) {
                        std::cout << \
                            "  Operator: " <<                            \
                            operators->at(i).metadata.operator_label <<  \
                            " / component: " << s + 1 <<                 \
                            " / origin: " <<                             \
                            operators->at(i).metadata.origin_label <<    \
                            std::endl;
                    }

                    // Wrappers over vectors.
                    arma::vec product_alph(products_alph[i].colptr(s), nov_alph, false, true);
                    arma::vec rspvec_alph(operators->at(i).rspvecs_alph.colptr(s), nov_alph, false, true);
                    arma::vec rspvec_old_alph(rspvecs_old_alph[i].colptr(s), nov_alph, false, true);
                    const arma::vec rhsvec_alph(operators->at(i).integrals_mo_ai_alph.colptr(s), nov_alph, false, true);

                    // The Armadillo vectors need to pass through to
                    // the correct memory locations, but the pointers
                    // are undefined for 1 density since the
                    // std::vectors are unpopulated.
                    double * product_beta_ptr = NULL;
                    double * rspvec_beta_ptr = NULL;
                    double * rspvec_old_beta_ptr = NULL;
                    double * rhsvec_beta_ptr = NULL;
                    if (nden == 2) {
                        product_beta_ptr = products_beta[i].colptr(s);
                        rspvec_beta_ptr = operators->at(i).rspvecs_beta.colptr(s);
                        rspvec_old_beta_ptr = rspvecs_old_beta[i].colptr(s);
                        rhsvec_beta_ptr = operators->at(i).integrals_mo_ai_beta.colptr(s);
                    }
                    // This is safe even with null pointers, because I'm a
                    // careful programmer...sometimes?
                    arma::vec product_beta(product_beta_ptr, nov_beta, false, true);
                    arma::vec rspvec_beta(rspvec_beta_ptr, nov_beta, false, true);
                    arma::vec rspvec_old_beta(rspvec_old_beta_ptr, nov_beta, false, true);
                    const arma::vec rhsvec_beta(rhsvec_beta_ptr, nov_beta, false, true);

                    // Perform the response vector update.
                    info.max_rmsd_alph = 0.0;
                    info.max_rmsd_beta = 0.0;
                    for (size_t iter = 0; iter < maxiter; iter++) {

                        if (print_level >= 10) {
                            rspvec_alph.print("rspvec_old_alph");
                            if (nden == 2)
                                rspvec_beta.print("rspvec_old_beta");
                        }

                        if (do_compute_generalized_density) {
                            // Compute J and K from D.
                            compute_generalized_density(Dg.slice(0), rspvec_alph, C_occ_alph, C_virt_alph);
                            if (nden == 2)
                                compute_generalized_density(Dg.slice(1), rspvec_beta, C_occ_beta, C_virt_beta);

                            if (print_level >= 10)
                                pretty_print(Dg, "Dg");

                            if (cfg->get_param<bool>("_mask_dg_ao")) {
                                make_masked_cube(Dg_masked, Dg, indices_ao, 0.0);
                                Dg = Dg_masked;
                                if (print_level >= 10)
                                    pretty_print(Dg, "Dg (masked)");
                            }

                            matvec->compute(J, K, Dg);
                        } else {
                            // Compute J and K from L and R.
                            // TODO implement index masking for L and R?
                            L[0] = C_virt_alph;
                            arma::mat qm_alph(rspvec_alph.memptr(), nvirt_alph, nocc_alph, false, true);
                            R[0] = (qm_alph * C_occ_alph.t()).t();
                            if (nden == 2) {
                                L[1] = C_virt_beta;
                                arma::mat qm_beta(rspvec_beta.memptr(), nvirt_beta, nocc_beta, false, true);
                                R[1] = (qm_beta * C_occ_beta.t()).t();
                            }
                            matvec->compute(J, K, L, R);
                        }

                        if (print_level >= 10) {
                            pretty_print(J, "J");
                            pretty_print(K, "K");
                        }

                        if (cfg->get_param<bool>("_mask_j_ao")) {
                            make_masked_cube(J_masked, J, indices_ao, 0.0);
                            J = J_masked;
                            if (print_level >= 10)
                                pretty_print(J, "J (masked)");
                        }
                        if (cfg->get_param<bool>("_mask_k_ao")) {
                            make_masked_cube(K_masked, K, indices_ao, 0.0);
                            K = K_masked;
                            if (print_level >= 10)
                                pretty_print(K, "K (masked)");
                        }

                        form_orbital_hessian_equations(ints_mnov, J, K, hamiltonian, spin, b_prefactor);

                        if (print_level >= 10)
                            pretty_print(ints_mnov, "ints_mnov");

                        if (cfg->get_param<bool>("_mask_ints_mnov_ao")) {
                            make_masked_cube(ints_mnov_masked, ints_mnov, indices_ao, 0.0);
                            ints_mnov = ints_mnov_masked;
                            if (print_level >= 10)
                                pretty_print(ints_mnov, "ints_mnov (masked)");
                        }

                        // There's no mask call for ints_ovov_spin since
                        // they get repacked into (product) vectors which
                        // can then be masked.
                        AO2MO(ints_ovov_alph, ints_mnov.slice(0), C_virt_alph, C_occ_alph);
                        if (nden == 2)
                            AO2MO(ints_ovov_beta, ints_mnov.slice(1), C_virt_beta, C_occ_beta);

                        repack_matrix_to_vector(product_alph, ints_ovov_alph);
                        if (nden == 2)
                            repack_matrix_to_vector(product_beta, ints_ovov_beta);

                        if (print_level >= 10) {
                            product_alph.print("product_alph");
                            if (nden == 2)
                                product_beta.print("product_beta");
                        }

                        if (cfg->get_param<bool>("_mask_product_mo")) {
                            arma::vec product_masked_alph(product_alph.n_elem, arma::fill::zeros);
                            product_masked_alph(indices_mo_alph) = product_alph(indices_mo_alph);
                            product_alph = product_masked_alph;
                            if (print_level >= 10)
                                product_alph.print("product_alph (masked)");
                            if (nden == 2) {
                                arma::vec product_masked_beta(product_beta.n_elem, arma::fill::zeros);
                                product_masked_beta(indices_mo_beta) = product_beta(indices_mo_beta);
                                product_beta = product_masked_beta;
                                if (print_level >= 10)
                                    product_beta.print("product_beta (masked)");
                            }
                        }

                        if (reduce) {

                            // shrink
                            // TODO rhsvec only needs to be done one time per operator component
                            rhsvec_reduced_alph = rhsvec_alph(indices_mo_alph);
                            product_reduced_alph = product_alph(indices_mo_alph);
                            rspvec_reduced_alph = rspvec_alph(indices_mo_alph);
                            if (nden == 2) {
                                rhsvec_reduced_beta = rhsvec_beta(indices_mo_beta);
                                product_reduced_beta = product_beta(indices_mo_beta);
                                rspvec_reduced_beta = rspvec_beta(indices_mo_beta);
                            }

                            // calculate
                            form_new_rspvec(rspvec_reduced_alph, product_reduced_alph, rhsvec_reduced_alph, *ediff_alph, frequency);
                            if (nden == 2)
                                form_new_rspvec(rspvec_reduced_beta, product_reduced_beta, rhsvec_reduced_beta, *ediff_beta, frequency);

                            // expand
                            // clean up to be safe
                            rspvec_alph.zeros();
                            rspvec_alph(indices_mo_alph) = rspvec_reduced_alph(indices_mo_red_alph);
                            if (nden == 2) {
                                rspvec_beta.zeros();
                                rspvec_beta(indices_mo_beta) = rspvec_reduced_beta(indices_mo_red_beta);
                            }

                        } else {

                            form_new_rspvec(rspvec_alph, product_alph, rhsvec_alph, *ediff_alph, frequency);
                            if (nden == 2)
                                form_new_rspvec(rspvec_beta, product_beta, rhsvec_beta, *ediff_beta, frequency);

                        }

                        if (print_level >= 10) {
                            rspvec_alph.print("rspvec_alph");
                            if (nden == 2)
                                rspvec_beta.print("rspvec_beta");
                        }

                        if (cfg->get_param<bool>("_mask_rspvec_mo")) {
                            arma::vec rspvec_masked_alph(rspvec_alph.n_elem, arma::fill::zeros);
                            rspvec_masked_alph(indices_mo_alph) = rspvec_alph(indices_mo_alph);
                            rspvec_alph = rspvec_masked_alph;
                            if (nden == 2) {
                                arma::vec rspvec_masked_beta(rspvec_beta.n_elem, arma::fill::zeros);
                                rspvec_masked_beta(indices_mo_beta) = rspvec_beta(indices_mo_beta);
                                rspvec_beta = rspvec_masked_beta;
                            }
                            if (print_level >= 10) {
                                rspvec_alph.print("rspvec_alph (masked)");
                                if (nden == 2)
                                    rspvec_beta.print("rspvec_beta (masked)");
                            }
                        }

                        // Compute and check for convergence.
                        info.curr_rmsd_alph = rmsd(rspvec_alph, rspvec_old_alph);
                        if (nden == 2) {
                            info.curr_rmsd_beta = rmsd(rspvec_beta, rspvec_old_beta);
                        }
                        info.iter = iter + 1;
                        info.s = s + 1;
                        if (print_level >= 2) {
                            std::cout << info << std::endl;
                        }
                        if (info.curr_rmsd_alph < conv) {
                            if (nden == 1) {
                                is_converged = true;
                                break;
                            } else if (info.curr_rmsd_beta < conv) {
                                is_converged = true;
                                break;
                            }
                        }

                        rspvec_old_alph = rspvec_alph;
                        if (nden == 2)
                            rspvec_old_beta = rspvec_beta;

                    }

                    // If not converged after the maximum number of
                    // iterations, crash.
                    if (!is_converged) {
                        throw std::runtime_error("not converged after max iterations");
                    }

                }

            }

        }

        return;

    }

};

} // namespace libresponse

#endif // LIBRESPONSE_LINEAR_ITERATOR_H_
