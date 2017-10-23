#include "dump_ao_integrals.h"

template void dump_integrals(
    const arma::mat& arma_integrals,
    const std::string& integral_description,
    const std::string& input_basename,
    const std::string& integral_filename_ending);
template void dump_integrals(
    const arma::cube& arma_integrals,
    const std::string& integral_description,
    const std::string& input_basename,
    const std::string& integral_filename_ending);

void dump_ao_integrals(const std::vector<libresponse::operator_spec> &operators, const std::string &basename)
{
    for (size_t i = 0; i < operators.size(); i++) {
        dump_integrals(
            operators[i].integrals_ao,
            operators[i].metadata.operator_label,
            basename,
            operators[i].metadata.operator_label + std::string(".dat"));
    }
}
