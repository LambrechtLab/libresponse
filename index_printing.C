#include "index_printing.h"

std::ostream& operator<<(std::ostream& os, const libresponse::type::indices& i)
{

    const size_t size = i.size();

    os << " size: " << size << std::endl;

    for (size_t s = 0; s < size; s++) {
        os << " block " << s << std::endl;
        os << i.at(s);
    }

    return os;

}

std::ostream& operator<<(std::ostream& os, const libresponse::type::pair_indices& pi)
{

    os << " pair (first)" << std::endl;
    os << pi.first;
    os << " pair (second)" << std::endl;
    os << pi.second;

    return os;

}

std::ostream& operator<<(std::ostream& os, const libresponse::type::pair_arma& pa)
{

    os << " pair (first)" << std::endl;
    os << pa.first;
    os << " pair (second)" << std::endl;
    os << pa.second;

    return os;

}

std::ostream& operator<<(std::ostream& os, const libresponse::type::pair& p)
{

    os << " (" << p.first << ", " << p.second << ")";

    return os;

}

std::ostream& operator<<(std::ostream& os, const libresponse::type::pairs& ps)
{

    libresponse::type::pairs_iterator it;

    for (it = ps.begin(); it != ps.end(); ++it) {
        os << *it << std::endl;
    }

    return os;

}

// template <typename T>
// std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {

//     typename std::vector<T>::const_iterator it;

//     os << "[";
//     for (it = v.begin(); it != v.end(); ++it) {
//         if (it == v.end() - 1)
//             os << *it;
//         else
//             os << *it << ", ";
//     }
//     os << "]" << std::endl;

//     return os;

// }

// Don't need explicit instantiation?
