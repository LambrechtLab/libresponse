#ifndef LIBRESPONSE_DUMP_AO_INTEGRALS_H_
#define LIBRESPONSE_DUMP_AO_INTEGRALS_H_

#include "operator_spec.h"

/*!
 * @brief Write the contents of an Armadillo container (matrix, cube)
 * to disk.
 */
template<typename T>
void dump_integrals(
    const T& arma_integrals,
    const std::string& integral_description,
    const std::string& input_basename,
    const std::string& integral_filename_ending) {

    const std::string integral_filename = \
        input_basename + "." + integral_filename_ending;

    std::cout                      \
        << "  Dumping "            \
        << integral_description    \
        << " integrals to file: "  \
        << integral_filename       \
        << std::endl;

    const bool res = arma_integrals.save(integral_filename, arma::arma_ascii);

    if (!res) {
        std::cout << "  Couldn't save integrals to file." << std::endl;
        throw 1;
    }

    return;

}

void dump_ao_integrals(const std::vector<libresponse::operator_spec> &operators, const std::string &basename);

#endif // LIBRESPONSE_DUMP_AO_INTEGRALS_H_
