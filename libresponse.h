/*!
 * @mainpage libresponse and responseman
 *
 * libresponse is a library for solving linear response equations over
 * arbitrary operators. responseman is a user-facing (input file)
 * wrapper around libresponse for Q-Chem.
 *
 * For an example of how to use the library from within Q-Chem, see
 * polman, which calculates the static polarizability, or magnetman,
 * which calculates open-shell magnetic properties (electronic
 * g-tensor, hyperfine tensors).
 *
 * Currently, documentation is missing in many places.
 *
 * @authors Eric Berquist 2016--present erb74 at pitt dot edu
 * @authors Evgeny Epifanovsky 2016
 */

#include "configurable.h"
#include "operator_spec.h"

/*!
 * Set default parameters for a solver.
 *
 * @param[out] &options option map
 */
void set_defaults(libresponse::configurable &options);
