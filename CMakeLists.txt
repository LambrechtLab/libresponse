cmake_minimum_required(VERSION 2.8.5)
project(LIBRESPONSE CXX)

find_package(Armadillo REQUIRED)

set(SRC
    configurable.C
    dump_ao_integrals.C
    index_printing.C
    indices.C
    matvec_i.C
    utils.C
    linear/helpers.C
    linear/interface.C
    linear/interface_nonorthogonal.C
    linear/printing.C
    operator_spec.C
    set_defaults.C
    )

add_library(response ${SRC})
target_link_libraries(response "${ARMADILLO_LIBRARIES}")

if(OPENMP_FOUND)
    set_target_properties(response PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)
