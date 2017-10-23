#include <iomanip>

#include "printing.h"

std::ostream& operator<<(std::ostream& os, const iteration_info_linear& info)
{

    os << "   iter: " << std::setw(9) << info.iter              \
       << " vec: " << std::setw(9) << info.s                            \
       << " curr_rmsd_alph: " << std::scientific << std::setw(12) << std::setprecision(6) << info.curr_rmsd_alph;
    if (info.has_beta) {
        os << " curr_rmsd_beta: " << std::scientific << std::setw(12) << std::setprecision(6) << info.curr_rmsd_beta;
    }

    return os;

}
