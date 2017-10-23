#ifndef LIBRESPONSE_LINEAR_PRINTING_H_
#define LIBRESPONSE_LINEAR_PRINTING_H_

#include <ostream>

struct iteration_info_linear {
    bool has_beta;
    size_t iter;
    size_t s;
    double curr_rmsd_alph;
    double curr_rmsd_beta;
    double max_rmsd_alph;
    double max_rmsd_beta;
};

std::ostream& operator<<(std::ostream& os, const iteration_info_linear& info);

#endif // LIBRESPONSE_LINEAR_PRINTING_H_
