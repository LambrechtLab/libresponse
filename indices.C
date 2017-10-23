#include <cassert>

#include "indices.h"
#include "utils.h"

namespace libresponse {

type::indices make_indices_ao(const arma::uvec &nbasis_frgm)
{

    const size_t nfrgm = nbasis_frgm.n_elem;
    type::indices v;
    size_t start, stop;
    for (size_t i = 0; i < nfrgm; i++) {
        if (i == 0)
            start = 0;
        else
            start = arma::accu(nbasis_frgm.subvec(0, i - 1));
        stop = start + nbasis_frgm(i);
        v.push_back(range(start, stop));
    }

    return v;

}

type::pair_indices make_indices_mo_separate(const arma::uvec &nocc_frgm, const arma::uvec &nvirt_frgm)
{

    if (nocc_frgm.n_elem != nvirt_frgm.n_elem)
        throw std::runtime_error("nocc_frgm.n_elem != nvirt_frgm.n_elem");
    const size_t nocc = arma::accu(nocc_frgm);
    const size_t nfrgm = nocc_frgm.n_elem;
    type::indices v_occ, v_virt;
    size_t start_occ, stop_occ, start_virt, stop_virt;
    for (size_t i = 0; i < nfrgm; i++) {
        if (i == 0) {
            start_occ = 0;
            start_virt = nocc;
        } else {
            start_occ = arma::accu(nocc_frgm.subvec(0, i - 1));
            start_virt = nocc + arma::accu(nvirt_frgm.subvec(0, i - 1));
        }
        stop_occ = start_occ + nocc_frgm(i);
        stop_virt = start_virt + nvirt_frgm(i);
        v_occ.push_back(range(start_occ, stop_occ));
        v_virt.push_back(range(start_virt, stop_virt));
    }

    type::pair_indices p = std::make_pair(v_occ, v_virt);

    return p;

}

type::indices make_indices_mo_combined(const arma::uvec &nocc_frgm, const arma::uvec &nvirt_frgm)
{

    type::indices v;
    const type::pair_indices p = make_indices_mo_separate(nocc_frgm, nvirt_frgm);
    const size_t nfrgm = nocc_frgm.n_elem;
    for (size_t i = 0; i < nfrgm; i++) {
        // force qualified name lookup
        // https://stackoverflow.com/a/7376212
        v.push_back(::join(p.first[i], p.second[i]));
    }

    return v;

}

arma::uvec make_indices_mo_restricted(const arma::uvec &nocc_frgm, const arma::uvec &nvirt_frgm)
{

    const size_t nocc_tot = arma::accu(nocc_frgm);
    const size_t nvirt_tot = arma::accu(nvirt_frgm);
    const size_t norb_tot = nocc_tot + nvirt_tot;
    const size_t nfrgm = nocc_frgm.n_elem;

    const type::pair_indices p = make_indices_mo_separate(nocc_frgm, nvirt_frgm);
    type::pairs pairs_all, pairs_good, pairs_bad;

    // Collect all possible occ-virt excitation pairs within each
    // fragment.
    for (size_t f = 0; f < nfrgm; f++) {
        for (size_t i = 0; i < p.first[f].n_elem; i++) {
            for (size_t a = 0; a < p.second[f].n_elem; a++) {
                pairs_good.insert(std::make_pair(p.first[f](i), p.second[f](a)));
            }
        }
    }

    // Collect all possible occ-virt excitation pairs for the
    // supersystem.
    for (size_t i = 0; i < nocc_tot; i++) {
        for (size_t a = nocc_tot; a < norb_tot; a++) {
            pairs_all.insert(std::make_pair(i, a));
        }
    }

    // All the "bad" (disallowed) pairs are the difference between the
    // supersystem and the fragment-localized pairs.

    // We do this by iterating over the set of all possible pairs, and
    // ones that are not members of the restricted set are added to
    // the disallowed set.

    type::pairs_iterator it_all;

    for (it_all = pairs_all.begin(); it_all != pairs_all.end(); ++it_all) {
        if (pairs_good.count(*it_all) == 0)
            pairs_bad.insert(*it_all);
    }

    if ((pairs_good.size() + pairs_bad.size()) != pairs_all.size())
        throw std::runtime_error("inconsistent sizes between good + bad and all pairs");

    // Now, convert all of the pairs to compound indices.
    std::vector<size_t> v_all, v_good, v_bad;
    size_t i = 0;
    for (it_all = pairs_all.begin(); it_all != pairs_all.end(); ++it_all) {
        v_all.push_back(i);
        if (pairs_good.count(*it_all) > 0)
            v_good.push_back(i);
        if (pairs_bad.count(*it_all) > 0)
            v_bad.push_back(i);
        i++;
    }

    // std::cout << "pairs_all" << std::endl;
    // std::cout << pairs_all << std::endl;
    // std::cout << "pairs_good" << std::endl;
    // std::cout << pairs_good << std::endl;
    // std::cout << "pairs_bad" << std::endl;
    // std::cout << pairs_bad << std::endl;
    // std::cout << "v_all" << std::endl;
    // std::cout << v_all << std::endl;
    // std::cout << "v_good" << std::endl;
    // std::cout << v_good << std::endl;
    // std::cout << "v_bad" << std::endl;
    // std::cout << v_bad << std::endl;

    return arma::conv_to<arma::uvec>::from(v_good);
}

type::indices make_indices_mo_restricted_local_occ_all_virt(const arma::uvec &nocc_frgm, const arma::uvec &nvirt_frgm)
{
    const size_t nocc_tot = arma::accu(nocc_frgm);
    const size_t nvirt_tot = arma::accu(nvirt_frgm);
    const size_t norb_tot = nocc_tot + nvirt_tot;
    const size_t nfrgm = nocc_frgm.n_elem;

    const type::pair_indices p = make_indices_mo_separate(nocc_frgm, nvirt_frgm);

    // For each fragment, collect pairs corresonding to its occupied
    // indices to all virtual indices (spanning all fragments).
    std::vector<type::pairs> pairs_per_frgm;
    for (size_t f = 0; f < nfrgm; f++) {
        type::pairs pairs_frgm;
        for (size_t i = 0; i < p.first[f].n_elem; i++) {
            for (size_t a = nocc_tot; a < norb_tot; a++) {
                pairs_frgm.insert(std::make_pair(p.first[f](i), a));
            }
        }
        pairs_per_frgm.push_back(pairs_frgm);
    }

    // Collect all possible occ-virt excitation pairs for the
    // supersystem.
    type::pairs pairs_all;
    for (size_t i = 0; i < nocc_tot; i++) {
        for (size_t a = nocc_tot; a < norb_tot; a++) {
            pairs_all.insert(std::make_pair(i, a));
        }
    }

    // Convert the allowed pairs per fragment into compound indices.
    type::indices v_all;
    type::pairs_iterator it_all;
    for (size_t f = 0; f < nfrgm; f++) {
        const type::pairs pairs_frgm = pairs_per_frgm.at(f);
        std::vector<size_t> v_frgm;
        size_t i = 0;
        for (it_all = pairs_all.begin(); it_all != pairs_all.end(); ++it_all) {
            if (pairs_frgm.count(*it_all) > 0) {
                // std::cout << i << " " << *it_all << std::endl;
                v_frgm.push_back(i);
            }
            i++;
        }
        v_all.push_back(arma::conv_to<arma::uvec>::from(v_frgm));
    }

    return v_all;
}

arma::uvec join(const type::indices &idxs)
{

    const size_t size = idxs.size();
    arma::uvec v;
    if (size == 0) {
    } else if (size == 1) {
        v = idxs[0];
    } else {
        // for (size_t i = 0; i < size; i++)
        //     v = ::join(v, idxs[i]);
        v = idxs[0];
        arma::uvec x;
        for (size_t i = 1; i < size; i++) {
            const arma::uvec w = idxs[i];
            const size_t lw = w.n_elem;
            const size_t lv = v.n_elem;
            x.set_size(lv + lw);
            x.subvec(0, lv - 1) = v;
            x.subvec(lv, lv + lw - 1) = w;
            v = x;
        }
    }

    return v;

}

void make_masked_mat(arma::mat &mm, const arma::mat &m, const arma::uvec &idxs, double fill_value, bool reduce)
{

    if (reduce) {

        const size_t dim = idxs.n_elem;
        mm.set_size(dim, dim);
        mm.fill(fill_value);

        mm = m.submat(idxs, idxs);

    } else {

        mm.set_size(m.n_rows, m.n_cols);
        mm.fill(fill_value);

        mm.submat(idxs, idxs) = m.submat(idxs, idxs);

    }

    return;

}

void make_masked_cube(arma::cube &mc, const arma::cube &c, const arma::uvec &idxs, double fill_value, bool reduce)
{

    if (reduce) {
        const size_t dim = idxs.n_elem;
        mc.set_size(dim, dim, c.n_slices);
    } else
        mc.set_size(c.n_rows, c.n_cols, c.n_slices);

    for (size_t ns = 0; ns < c.n_slices; ns++) {
        make_masked_mat(mc.slice(ns), c.slice(ns), idxs, fill_value, reduce);
    }

    return;

}

// fill then copy, rather than copy then fill
void make_masked_mat(arma::mat &mm, const arma::mat &m, const type::indices &idxs, double fill_value, bool reduce)
{

    if (idxs.empty())
        throw std::runtime_error("idxs.empty()");

    const size_t nblocks = idxs.size();

    if (reduce) {

        const arma::uvec idxs_joined = join(idxs);
        const size_t dim = idxs_joined.n_elem;

        mm.set_size(dim, dim);
        mm.fill(fill_value);

        mm = m.submat(idxs_joined, idxs_joined);

    } else {

        mm.set_size(m.n_rows, m.n_cols);
        mm.fill(fill_value);

        for (size_t i = 0; i < nblocks; i++) {
            // submat takes 2 uvecs and automatically forms the correct
            // outer product between them for indexing
            mm.submat(idxs[i], idxs[i]) = m.submat(idxs[i], idxs[i]);
        }

    }

    return;

}

// TODO template over index type?
void make_masked_cube(arma::cube &mc, const arma::cube &c, const type::indices &idxs, double fill_value, bool reduce)
{

    if (reduce) {
        // urgh, my kingdom for a list comprehension
        const size_t dim = join(idxs).n_elem;
        mc.set_size(dim, dim, c.n_slices);
    } else
        mc.set_size(c.n_rows, c.n_cols, c.n_slices);

    for (size_t ns = 0; ns < c.n_slices; ns++) {
        make_masked_mat(mc.slice(ns), c.slice(ns), idxs, fill_value, reduce);
    }

    return;

}

void make_masked_mat(arma::mat &mm, const arma::mat &m, const type::indices &idxs_rows, const type::indices &idxs_cols, double fill_value, bool reduce)
{

    if (idxs_rows.empty() || idxs_cols.empty())
        throw std::runtime_error("idxs_rows.empty() || idxs_cols.empty()");

    if (reduce) {

        const arma::uvec idxs_rows_joined = join(idxs_rows);
        const size_t dim_rows = idxs_rows_joined.n_elem;
        const arma::uvec idxs_cols_joined = join(idxs_cols);
        const size_t dim_cols = idxs_cols_joined.n_elem;

        mm.set_size(dim_rows, dim_cols);
        mm.fill(fill_value);

        mm = m.submat(idxs_rows_joined, idxs_cols_joined);

    } else {

        mm.set_size(m.n_rows, m.n_cols);
        mm.fill(fill_value);

        // this is an artificial constraint, there's probably a better way
        // of doing this
        if (idxs_rows.size() != idxs_cols.size())
            throw std::runtime_error("idxs_rows.size() != idxs_cols.size()");

        const size_t nblocks = idxs_rows.size();

        for (size_t i = 0; i < nblocks; i++) {
            // submat takes 2 uvecs and automatically forms the correct
            // outer product between them for indexing
            mm.submat(idxs_rows[i], idxs_cols[i]) = m.submat(idxs_rows[i], idxs_cols[i]);
        }

    }

    return;

}

void make_masked_cube(arma::cube &mc, const arma::cube &c, const type::indices &idxs_rows, const type::indices &idxs_cols, double fill_value, bool reduce)
{

    if (reduce) {
        const size_t dim_rows = join(idxs_rows).n_elem;
        const size_t dim_cols = join(idxs_cols).n_elem;
        mc.set_size(dim_rows, dim_cols, c.n_slices);
    } else
        mc.set_size(c.n_rows, c.n_cols, c.n_slices);

    for (size_t ns = 0; ns < c.n_slices; ns++) {
        make_masked_mat(mc.slice(ns), c.slice(ns), idxs_rows, idxs_cols, fill_value, reduce);
    }

    return;

}

template <typename T>
std::vector<T> set_to_ordered_vector(const std::set<T> &s)
{

    std::vector<T> v(s.begin(), s.end());
    std::sort(v.begin(), v.end());

    return v;

}

template std::vector<size_t> set_to_ordered_vector(const std::set<size_t> &s);

type::pair_arma make_indices_from_mask(const arma::umat &mask, int mask_val_for_return)
{

    // blow up to avoid weird casting tricks
    if (mask_val_for_return < 0 || mask_val_for_return > 1)
        throw std::runtime_error("mask_val_for_return < 0 || mask_val_for_return > 1");

    std::set<size_t> sr, sc;

    for (size_t i = 0; i < mask.n_rows; i++) {
        for (size_t j = 0; j < mask.n_cols; j++) {
            if (mask(i, j) == mask_val_for_return) {
                sr.insert(i);
                sc.insert(j);
            }
        }
    }

    // transfer the set contents to vectors, which are then sorted in
    // place
    const std::vector<size_t> vr = set_to_ordered_vector(sr);
    const std::vector<size_t> vc = set_to_ordered_vector(sc);

    const arma::uvec ar = arma::conv_to<arma::uvec>::from(vr);
    const arma::uvec ac = arma::conv_to<arma::uvec>::from(vc);

    return std::make_pair(ar, ac);

}

} // namespace libresponse
