#include <cassert>

#include "../indices.h"
#include "../operator_spec.h"
#include "../utils.h"

namespace libresponse {

void one_electron_mn_mats_to_ia_vecs(
    arma::mat &ia_vecs,
    const arma::cube &mn_mats,
    const arma::mat &C_occ,
    const arma::mat &C_virt)
{

    const size_t nocc = C_occ.n_cols;
    const size_t nvirt = C_virt.n_cols;
    const size_t nov = nocc * nvirt;

    const size_t n_slices = mn_mats.n_slices;

    assert(ia_vecs.n_rows == nov);
    assert(ia_vecs.n_cols == n_slices);

    arma::cube ia_mats(nvirt, nocc, n_slices);
    // munu -> ai
    for (size_t vs = 0; vs < n_slices; vs++)
        AO2MO(ia_mats.slice(vs), mn_mats.slice(vs), C_virt, C_occ);

    for (size_t s = 0; s < n_slices; s++) {
        arma::vec ia_vec(ia_vecs.colptr(s), nov, false, false);
        repack_matrix_to_vector(ia_vec, ia_mats.slice(s));
    }

    return;

}

void one_electron_ia_vecs_to_mn_mats(
    arma::cube &mn_mats,
    const arma::mat &ia_vecs,
    const arma::mat &C_occ,
    const arma::mat &C_virt)
{

    const size_t nocc = C_occ.n_cols;
    const size_t nvirt = C_virt.n_cols;
    const size_t nov = nocc * nvirt;

    const size_t n_slices = mn_mats.n_slices;

    assert(ia_vecs.n_rows == nov);
    assert(ia_vecs.n_cols == n_slices);

    arma::cube ia_mats(nvirt, nocc, n_slices);
    for (size_t vs = 0; vs < n_slices; vs++) {
        const arma::vec ia_vec(const_cast<double *>(ia_vecs.colptr(vs)), nov, false, false);
        repack_vector_to_matrix(ia_mats.slice(vs), ia_vec);
    }

    // ai -> munu
    for (size_t s = 0; s < n_slices; s++)
        mn_mats.slice(s) = C_virt * ia_mats.slice(s) * C_occ.t();

    return;

}

void one_electron_mn_mats_to_ia_vecs(
    arma::mat &ia_vecs,
    const arma::cube &mn_mats,
    const arma::mat &C_occ,
    const arma::mat &C_virt,
    const type::indices &mask_indices)
{

    const size_t nocc = C_occ.n_cols;
    const size_t nvirt = C_virt.n_cols;
    const size_t nov = nocc * nvirt;

    const size_t n_slices = mn_mats.n_slices;

    // TODO add more asserts?
    assert(ia_vecs.n_rows == nov);
    assert(ia_vecs.n_cols == n_slices);

    arma::cube ia_mats(nvirt, nocc, n_slices);
    arma::mat mn_mat_masked;
    for (size_t vs = 0; vs < n_slices; vs++) {
        // mn_mats.slice(vs).print("AO before mask");
        make_masked_mat(mn_mat_masked, mn_mats.slice(vs), mask_indices, 0.0);
        // mn_mat_masked.print("AO after mask");
        AO2MO(ia_mats.slice(vs), mn_mat_masked, C_virt, C_occ);
        // ia_mats.slice(vs).print("MO");
    }

    for (size_t s = 0; s < n_slices; s++) {
        arma::vec ia_vec(ia_vecs.colptr(s), nov, false, false);
        repack_matrix_to_vector(ia_vec, ia_mats.slice(s));
        // ia_vec.print("MO vec");
    }

    return;

}

void form_vec_energy_differences(
    arma::vec &ediff,
    const arma::vec &moene_i,
    const arma::vec &moene_a
    )
{

    const size_t nocc = moene_i.n_elem;
    const size_t nvirt = moene_a.n_elem;

    assert(ediff.n_elem == (nocc * nvirt));

    size_t ia;

    for (size_t i = 0; i < nocc; i++) {
        for (size_t a = 0; a < nvirt; a++) {
            ia = i*nvirt + a;
            ediff(ia) = moene_a(a) - moene_i(i);
        }
    }

    return;

}

void compute_generalized_density(
    arma::mat &Dg,
    const arma::vec &q,
    const arma::mat &C_occ,
    const arma::mat &C_virt
    )
{

    const size_t nocc = C_occ.n_cols;
    const size_t nvirt = C_virt.n_cols;

    // matrix view onto vector with compound index
    const arma::mat qm(const_cast<double *>(q.memptr()), nvirt, nocc, false, true);

    Dg = C_virt * qm * C_occ.t();

    return;

}

void form_guess_rspvec(
    arma::vec &rspvec,
    const arma::vec &rhsvec,
    const arma::vec &ediff,
    double frequency
    )
{

    assert(rspvec.n_elem == rhsvec.n_elem);
    assert(rspvec.n_elem == ediff.n_elem);

    rspvec = rhsvec / (ediff - frequency);

    return;

}

// TODO have a version that doesn't take a frequency that will
// take the inverse, so arma::solve is never called
void form_guess_rspvec(
    arma::vec &rspvec,
    const arma::vec &rhsvec,
    const arma::mat &ediff,
    double frequency
    )
{

    assert(rspvec.n_elem == rhsvec.n_elem);
    assert(rspvec.n_elem == ediff.n_rows);
    assert(ediff.n_rows == ediff.n_cols);

    // TODO what kind of options to pass to arma::solve?
    rspvec = arma::solve(ediff - (frequency * arma::eye<arma::mat>(ediff.n_rows, ediff.n_cols)), rhsvec);

    // std::cout << "in form_guess_rspvec" << std::endl;
    // arma::mat ediff_inv = arma::inv(ediff - (frequency * arma::eye<arma::mat>(ediff.n_rows, ediff.n_cols)));
    // (ediff_inv * rhsvec).print("rspvec from inv");
    // rspvec.print("rspvec from solve");
    // ediff_inv.print("ediff_inv");

    return;

}

void form_new_rspvec(
    arma::vec &rspvec,
    const arma::vec &product,
    const arma::vec &rhsvec,
    const arma::vec &ediff,
    double frequency
    )
{

    assert(rspvec.n_elem == product.n_elem);
    assert(rspvec.n_elem == rhsvec.n_elem);
    assert(rspvec.n_elem == ediff.n_elem);

    rspvec = (rhsvec - product) / (ediff - frequency);

    return;

}

// TODO have a version that doesn't take a frequency that will
// take the inverse, so arma::solve is never called
void form_new_rspvec(
    arma::vec &rspvec,
    const arma::vec &product,
    const arma::vec &rhsvec,
    const arma::mat &ediff,
    double frequency
    )
{

    assert(rspvec.n_elem == product.n_elem);
    assert(rspvec.n_elem == rhsvec.n_elem);
    assert(rspvec.n_elem == ediff.n_rows);
    assert(ediff.is_square());

    // TODO what kind of options to pass to arma::solve?
    rspvec = arma::solve(ediff - (frequency * arma::eye<arma::mat>(ediff.n_rows, ediff.n_cols)), rhsvec - product);

    return;

}

void form_results(
    arma::mat &results,
    const arma::mat &vecs_property,
    const arma::mat &vecs_response
    )
{

    assert(vecs_property.n_rows == vecs_response.n_rows);
    assert(vecs_property.n_cols == vecs_response.n_cols);
    assert(results.n_rows == vecs_property.n_cols);
    assert(results.n_cols == vecs_response.n_cols);

    for (size_t r = 0; r < vecs_property.n_cols; r++)
        for (size_t s = 0; s < vecs_response.n_cols; s++)
            results(r, s) = arma::dot(vecs_property.col(r), vecs_response.col(s));

    return;

}

void form_results(
    arma::mat &results,
    const arma::mat &vecs_property,
    const arma::mat &vecs_response,
    const arma::uvec &indices_mo
    )
{

    assert(vecs_property.n_rows == vecs_response.n_rows);
    assert(vecs_property.n_cols == vecs_response.n_cols);
    assert(results.n_rows == vecs_property.n_cols);
    assert(results.n_cols == vecs_response.n_cols);
    // ...
    assert(indices_mo.n_elem <= vecs_property.n_rows);
    assert(indices_mo.n_elem <= vecs_response.n_rows);

    arma::uvec cr(1);
    arma::uvec cs(1);
    for (size_t r = 0; r < vecs_property.n_cols; r++) {
        cr(0) = r;
        for (size_t s = 0; s < vecs_response.n_cols; s++) {
            cs(0) = s;
            results(r, s) = arma::dot(vecs_property.submat(indices_mo, cr),
                                      vecs_response.submat(indices_mo, cs));
        }
    }

    return;

}

void form_results(
    arma::mat &results,
    const std::vector<arma::mat> &vecs_property,
    const std::vector<arma::mat> &vecs_response
    )
{

    // Form the result blocks between each pair of operators.
    std::vector<arma::mat> result_blocks;
    std::vector<size_t> row_starts;
    std::vector<size_t> col_starts;
    size_t row_start, row_end, col_start, col_end;
    row_start = 0;
    for (size_t i = 0; i < vecs_property.size(); i++) {
        arma::mat vecs_property_operator(const_cast<double *>(vecs_property[i].memptr()),
                                         vecs_property[i].n_rows,
                                         vecs_property[i].n_cols,
                                         true, false);
        col_start = 0;
        for (size_t j = 0; j < vecs_response.size(); j++) {
            arma::mat vecs_response_operator(const_cast<double *>(vecs_response[j].memptr()),
                                             vecs_response[j].n_rows,
                                             vecs_response[j].n_cols,
                                             true, false);
            arma::mat results_ij(vecs_property_operator.n_cols, vecs_response_operator.n_cols);
            form_results(results_ij, vecs_property_operator, vecs_response_operator);
            result_blocks.push_back(results_ij);
            row_starts.push_back(row_start);
            col_starts.push_back(col_start);
            col_start += vecs_response_operator.n_cols;
        }
        row_start += vecs_property_operator.n_cols;
    }

    // Put each of the result blocks back in the main results
    // matrix.
    for (size_t i = 0; i < vecs_property.size(); i++) {
        for (size_t j = 0; j < vecs_response.size(); j++) {
            const size_t vec_offset = (i * vecs_response.size()) + j;
            arma::mat block(result_blocks[vec_offset].memptr(),
                            result_blocks[vec_offset].n_rows,
                            result_blocks[vec_offset].n_cols, true, false);
            row_start = row_starts[vec_offset];
            col_start = col_starts[vec_offset];
            row_end = row_start + block.n_rows - 1;
            col_end = col_start + block.n_cols - 1;
            results(arma::span(row_start, row_end), arma::span(col_start, col_end)) = block;
        }
    }

    return;

}

// TODO this smells like duplicate code
void form_results(
    arma::mat &results,
    const std::vector<arma::mat> &vecs_property,
    const std::vector<arma::mat> &vecs_response,
    const arma::uvec &indices_mo
    )
{

    // Form the result blocks between each pair of operators.
    std::vector<arma::mat> result_blocks;
    std::vector<size_t> row_starts;
    std::vector<size_t> col_starts;
    size_t row_start, row_end, col_start, col_end;
    row_start = 0;
    for (size_t i = 0; i < vecs_property.size(); i++) {
        arma::mat vecs_property_operator(const_cast<double *>(vecs_property[i].memptr()),
                                         vecs_property[i].n_rows,
                                         vecs_property[i].n_cols,
                                         true, false);
        col_start = 0;
        for (size_t j = 0; j < vecs_response.size(); j++) {
            arma::mat vecs_response_operator(const_cast<double *>(vecs_response[j].memptr()),
                                             vecs_response[j].n_rows,
                                             vecs_response[j].n_cols,
                                             true, false);
            arma::mat results_ij(vecs_property_operator.n_cols, vecs_response_operator.n_cols);
            form_results(results_ij, vecs_property_operator, vecs_response_operator, indices_mo);
            result_blocks.push_back(results_ij);
            row_starts.push_back(row_start);
            col_starts.push_back(col_start);
            col_start += vecs_response_operator.n_cols;
        }
        row_start += vecs_property_operator.n_cols;
    }

    // Put each of the result blocks back in the main results
    // matrix.
    for (size_t i = 0; i < vecs_property.size(); i++) {
        for (size_t j = 0; j < vecs_response.size(); j++) {
            const size_t vec_offset = (i * vecs_response.size()) + j;
            arma::mat block(result_blocks[vec_offset].memptr(),
                            result_blocks[vec_offset].n_rows,
                            result_blocks[vec_offset].n_cols, true, false);
            row_start = row_starts[vec_offset];
            col_start = col_starts[vec_offset];
            row_end = row_start + block.n_rows - 1;
            col_end = col_start + block.n_cols - 1;
            results(arma::span(row_start, row_end), arma::span(col_start, col_end)) = block;
        }
    }

    return;

}

void form_results(
    arma::cube &results,
    const std::vector<operator_spec> &operators,
    const type::indices * indices_mo) {

    const bool has_beta = (results.n_slices == 2);

    // Form the result blocks between each pair of operators.
    std::vector<arma::mat> result_blocks_alph, result_blocks_beta;
    std::vector<size_t> row_starts, col_starts;
    std::vector<size_t> indices_property_operators, indices_response_operators;
    size_t row_start, row_end, col_start, col_end;
    row_start = 0;
    for (size_t i = 0; i < operators.size(); i++) {
        const operator_spec osi = operators.at(i);
        arma::mat vecs_property_alph = osi.integrals_mo_ai_alph;
        arma::mat vecs_property_beta;
        if (has_beta)
            vecs_property_beta = osi.integrals_mo_ai_beta;
        col_start = 0;
        for (size_t j = 0; j < operators.size(); j++) {
            const operator_spec osj = operators.at(j);
            if (osj.do_response) {
                arma::mat vecs_response_alph = osj.rspvecs_alph;
                arma::mat vecs_response_beta;
                if (has_beta)
                    vecs_response_beta = osj.rspvecs_beta;
                arma::mat results_ij_alph(vecs_property_alph.n_cols, vecs_response_alph.n_cols);
                if (indices_mo)
                    form_results(results_ij_alph, vecs_property_alph, vecs_response_alph, indices_mo->at(0));
                else
                    form_results(results_ij_alph, vecs_property_alph, vecs_response_alph);
                // results_ij_alph.print("results_ij_alph");
                result_blocks_alph.push_back(results_ij_alph);
                if (has_beta) {
                    arma::mat results_ij_beta(vecs_property_beta.n_cols, vecs_response_beta.n_cols);
                    if (indices_mo)
                        form_results(results_ij_beta, vecs_property_beta, vecs_response_beta, indices_mo->at(1));
                    else
                        form_results(results_ij_beta, vecs_property_beta, vecs_response_beta);
                    result_blocks_beta.push_back(results_ij_beta);
                }
                row_starts.push_back(row_start);
                col_starts.push_back(col_start);
                col_start += vecs_response_alph.n_cols;
                indices_response_operators.push_back(j);
            }
        }
        row_start += vecs_property_alph.n_cols;
        indices_property_operators.push_back(i);
    }

    if (has_beta)
        assert(result_blocks_alph.size() == result_blocks_beta.size());
    else
        assert(result_blocks_beta.size() == 0);

    // Put each of the result blocks back in the main results
    // matrix.
    for (size_t irb = 0; irb < result_blocks_alph.size(); irb++) {
        arma::mat block_alph(result_blocks_alph[irb]);
        arma::mat block_beta;
        if (has_beta)
            block_beta = arma::mat(result_blocks_beta[irb]);
        row_start = row_starts[irb];
        col_start = col_starts[irb];
        row_end = row_start + block_alph.n_rows - 1;
        col_end = col_start + block_alph.n_cols - 1;
        results(arma::span(row_start, row_end), arma::span(col_start, col_end), arma::span(0)) = block_alph;
        if (has_beta)
            results(arma::span(row_start, row_end), arma::span(col_start, col_end), arma::span(1)) = block_beta;
    }
}

arma::umat occupations_to_ranges(const arma::uvec &occupations)
{

    arma::umat ranges(4, 2);

    const size_t nocc_alph = occupations(0);
    const size_t nvirt_alph = occupations(1);
    const size_t nocc_beta = occupations(2);
    const size_t nvirt_beta = occupations(3);

    ranges(0, 0) = 0;
    ranges(1, 0) = nocc_alph;
    ranges(2, 0) = nocc_alph;
    ranges(3, 0) = nvirt_alph + nocc_alph;
    ranges(0, 1) = 0;
    ranges(1, 1) = nocc_beta;
    ranges(2, 1) = nocc_beta;
    ranges(3, 1) = nvirt_beta + nocc_beta;

    return ranges;

}

arma::umat pack_fragment_occupations(
    const arma::uvec &nbasis_frgm,
    const arma::uvec &norb_frgm,
    const arma::uvec &nocc_frgm_alph,
    const arma::uvec &nocc_frgm_beta
    )
{

    assert(nbasis_frgm.n_elem == norb_frgm.n_elem);
    assert(nbasis_frgm.n_elem == nocc_frgm_alph.n_elem);
    assert(nbasis_frgm.n_elem == nocc_frgm_beta.n_elem);

    const size_t nfrgm = nbasis_frgm.n_elem;

    arma::umat fragment_occupations(nfrgm, 4);

    fragment_occupations.col(0) = nbasis_frgm;
    fragment_occupations.col(1) = norb_frgm;
    fragment_occupations.col(2) = nocc_frgm_alph;
    fragment_occupations.col(3) = nocc_frgm_beta;

    return fragment_occupations;

}

arma::cube check_fragment_locality(const arma::cube &mocoeffs,
                                   const arma::umat &fragment_occupations)
{

    const size_t nden = mocoeffs.n_slices;

    if (fragment_occupations.n_cols != 4)
        throw 1;
    const size_t nfrgm = fragment_occupations.n_rows;
    const arma::uvec nbasis_frgm = fragment_occupations.col(0);
    const arma::uvec norb_frgm = fragment_occupations.col(1);
    const size_t norb_tot = arma::accu(norb_frgm);
    if (norb_tot != mocoeffs.n_cols)
        throw 1;
    libresponse::type::indices indices_ao_all = libresponse::make_indices_ao(nbasis_frgm);

    arma::cube weights(norb_tot, nfrgm, nden, arma::fill::zeros);

    for (size_t s = 0; s < nden; s++) {
        for (size_t p = 0; p < norb_tot; p++) {
            for (size_t f = 0; f < nfrgm; f++) {
                const arma::uvec indices_ao_frgm = indices_ao_all[f];
                for (size_t i = 0; i < indices_ao_frgm.n_elem; i++) {
                    weights(p, f, s) += std::pow(mocoeffs(indices_ao_frgm(i), p, s), 2);
                }
            }
        }
    }

    return weights;

}

arma::cube weight_to_pct(const arma::cube &weights)
{

    // n_rows -> # of MOs
    // n_cols -> # of fragments
    // n_slices -> alpha/beta spin
    const size_t n_rows = weights.n_rows;
    const size_t n_cols = weights.n_cols;
    const size_t n_slices = weights.n_slices;

    // 6.600.x+
    // const arma::mat tot = arma::sum(weights, 1);
    arma::mat tot(n_rows, n_slices);

    arma::cube pcts(n_rows, n_cols, n_slices);

    for (size_t c = 0; c < n_slices; c++) {
        tot.col(c) = arma::sum(weights.slice(c), 1);
        for (size_t b = 0; b < n_cols; b++) {
            pcts.slice(c).col(b) = weights.slice(c).col(b) / tot.col(c);
        }
    }

    return pcts * 100;

}

void form_orbital_hessian_equations(
    arma::cube &orbhess,
    const arma::cube &J,
    const arma::cube &K,
    const std::string &hamiltonian,
    const std::string &spin,
    int b_prefactor)
{

    const size_t nbasis = orbhess.n_rows;
    assert(orbhess.n_cols == nbasis);
    assert(J.n_rows == nbasis);
    assert(J.n_cols == nbasis);
    assert(K.n_rows == nbasis);
    assert(K.n_cols == nbasis);
    const size_t nden = orbhess.n_slices;
    assert(nden == 1 || nden == 2);
    assert(J.n_slices == nden);
    assert(K.n_slices == nden);
    assert(b_prefactor == 1 || b_prefactor == -1);

    // These are the old, more concise forms of the equations.

    // if (nden == 1) {
    //     if (hamiltonian == "rpa" && spin == "singlet") {
    //         orbhess = 4*J - K.t() - K;
    //     } else if (hamiltonian == "rpa" && spin == "triplet") {
    //         orbhess = - K - K.t();
    //     } else if (hamiltonian == "tda" && spin == "singlet") {
    //         orbhess = 2*J - K;
    //     } else if (hamiltonian == "tda" && spin == "triplet") {
    //         orbhess = - K;
    //     } else {
    //         // throw exception
    //     }
    // }
    // if (nden == 2) {
    //     if (hamiltonian == "rpa" && spin == "singlet") {
    //         orbhess_alph = 2*(J_alph + J_beta) - (K_alph.t() + K_alph);
    //         orbhess_beta = 2*(J_beta + J_alph) - (K_beta.t() + K_beta);
    //     } else if (hamiltonian == "rpa" && spin == "triplet") {
    //         orbhess_alph = - K_alph - K_alph.t();
    //         orbhess_beta = - K_beta - K_beta.t();
    //     } else if (hamiltonian == "tda" && spin == "singlet") {
    //         orbhess_alph = (J_alph + J_beta) - K_alph;
    //         orbhess_beta = (J_beta + J_alph) - K_beta;
    //     } else if (hamiltonian == "tda" && spin == "triplet") {
    //         orbhess_alph = - K_alph;
    //         orbhess_beta = - K_beta;
    //     } else {
    //         // throw exception
    //     }
    // }

    if (nden == 1) {
        if (spin == "singlet") {
            // Always form the A matrix, needed for both TDA/RPA.
            orbhess.slice(0) = 2*J.slice(0) - K.slice(0);
            // B matrix contribution.
            if (hamiltonian == "rpa")
                // Only add the B matrix if doing RPA. 1, -1 ->
                // (A+B), (A-B) is for pure/imaginary
                // perturbations (also known as electric/magnetic
                // Hessians).
                orbhess.slice(0) += (2*J.slice(0) - K.slice(0).t()) * b_prefactor;
            else
                if (hamiltonian != "tda")
                    throw std::runtime_error("hamiltonian != rpa or tda");
        } else if (spin == "triplet") {
            orbhess.slice(0) = - K.slice(0);
            if (hamiltonian == "rpa")
                orbhess.slice(0) += (- K.slice(0).t()) * b_prefactor;
            else
                if (hamiltonian != "tda")
                    throw std::runtime_error("hamiltonian != rpa or tda");
        } else {
            throw std::runtime_error("spin != singlet or triplet");
        }
    }

    else if (nden == 2) {
        if (spin == "singlet") {
            orbhess.slice(0) = (J.slice(0) + J.slice(1)) - K.slice(0);
            orbhess.slice(1) = (J.slice(1) + J.slice(0)) - K.slice(1);
            if (hamiltonian == "rpa") {
                orbhess.slice(0) += ((J.slice(0) + J.slice(1)) - K.slice(0).t()) * b_prefactor;
                orbhess.slice(1) += ((J.slice(1) + J.slice(0)) - K.slice(1).t()) * b_prefactor;
            } else
                if (hamiltonian != "tda")
                    throw std::runtime_error("hamiltonian != rpa or tda");
        } else if (spin == "triplet") {
            orbhess.slice(0) = - K.slice(0);
            orbhess.slice(1) = - K.slice(1);
            if (hamiltonian == "rpa") {
                orbhess.slice(0) += (- K.slice(0).t()) * b_prefactor;
                orbhess.slice(1) += (- K.slice(1).t()) * b_prefactor;
            } else
                if (hamiltonian != "tda")
                    throw std::runtime_error("hamiltonian != rpa or tda");
        } else {
            throw std::runtime_error("spin != singlet or triplet");
        }
    }

    else {
        throw std::runtime_error("nden != 1 or 2");
    }

    // no B prefactor!
    // if (nden == 1) {
    //     if (hamiltonian == "rpa" && spin == "singlet") {
    //         orbhess.slice(0) = 4*J.slice(0) - K.slice(0).t() - K.slice(0);
    //     } else if (hamiltonian == "rpa" && spin == "triplet") {
    //         orbhess.slice(0) = - K.slice(0) - K.slice(0).t();
    //     } else if (hamiltonian == "tda" && spin == "singlet") {
    //         orbhess.slice(0) = 2*J.slice(0) - K.slice(0);
    //     } else if (hamiltonian == "tda" && spin == "triplet") {
    //         orbhess.slice(0) = - K.slice(0);
    //     } else {
    //         throw 1;
    //     }
    // }
    // else if (nden == 2) {
    //     if (hamiltonian == "rpa" && spin == "singlet") {
    //         orbhess.slice(0) = 2*(J.slice(0) + J.slice(1)) - (K.slice(0).t() + K.slice(0));
    //         orbhess.slice(1) = 2*(J.slice(1) + J.slice(0)) - (K.slice(1).t() + K.slice(1));
    //     } else if (hamiltonian == "rpa" && spin == "triplet") {
    //         orbhess.slice(0) = - K.slice(0) - K.slice(0).t();
    //         orbhess.slice(1) = - K.slice(1) - K.slice(1).t();
    //     } else if (hamiltonian == "tda" && spin == "singlet") {
    //         orbhess.slice(0) = (J.slice(0) + J.slice(1)) - K.slice(0);
    //         orbhess.slice(1) = (J.slice(1) + J.slice(0)) - K.slice(1);
    //     } else if (hamiltonian == "tda" && spin == "triplet") {
    //         orbhess.slice(0) = - K.slice(0);
    //         orbhess.slice(1) = - K.slice(1);
    //     } else {
    //         throw 1;
    //     }
    // }
    // else
    //     throw 1;

    return;

}

void test_idempotency(const arma::mat &M, const arma::mat &S)
{

    arma::mat diff = M*S*M - M;

    const double thresh = 1.0e-15;
    const double val = arma::accu(arma::abs(diff));
    std::cout << "idempotency check, M*S*M - M: " << val << std::endl;
    assert(val < thresh);

    return;

}

void form_ediff_terms(
    arma::mat &ediff_mat,
    const arma::mat &F,
    const arma::mat &S,
    size_t nocc,
    size_t nvirt
    )
{

    const size_t norb = nocc + nvirt;
    const size_t nov = nocc * nvirt;

    // norb, not nbasis, because we are in the nonorthogonal MO
    // basis rather than the AO basis.
    assert(S.n_rows == norb);
    assert(S.n_cols == norb);
    assert(F.n_rows == norb);
    assert(F.n_cols == norb);

    assert(ediff_mat.is_square());
    assert(ediff_mat.n_rows == nov);
    assert(ediff_mat.n_cols == nov);

    size_t i, a, j, b, ia, jb;
    for (i = 0; i < nocc; i++) {
        for (a = nocc; a < norb; a++) {
            ia = i*nvirt + a - nocc;
            for (j = 0; j < nocc; j++) {
                for (b = nocc; b < norb; b++) {
                    jb = j*nvirt + b - nocc;
                    ediff_mat(ia, jb) = (F(a, b) * S(i, j)) - (F(i, j) * S(a, b));
                }
            }
        }
    }

    return;

}

void form_superoverlap(
    arma::mat &superoverlap,
    const arma::mat &S,
    size_t nocc,
    size_t nvirt
    )
{

    const size_t norb = nocc + nvirt;
    const size_t nov = nocc * nvirt;

    // norb, not nbasis, because we are in the nonorthogonal
    // MO basis rather than the AO basis.
    assert(S.n_rows == norb);
    assert(S.n_cols == norb);

    assert(superoverlap.is_square());
    assert(superoverlap.n_rows == nov);
    assert(superoverlap.n_cols == nov);

    size_t i, a, j, b, ia, jb;
    for (i = 0; i < nocc; i++) {
        for (a = nocc; a < norb; a++) {
            ia = i*nvirt + a - nocc;
            for (j = 0; j < nocc; j++) {
                for (b = nocc; b < norb; b++) {
                    jb = j*nvirt + b - nocc;
                    superoverlap(ia, jb) = S(i, j) * S(a, b);
                }
            }
        }
    }

    return;

}

} // namespace libresponse
