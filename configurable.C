#include <iostream>
#include "configurable.h"

namespace libresponse {

const std::string &configurable::get_param(const std::string &key) const {
    ssmap_it_type i = m_params.find(key);
    return i == m_params.end() ? m_empty : i->second;
}

void configurable::cfg(const std::string &key, const std::string &val) {
    m_params[key] = val;
}

void configurable::dump_params() const {
    for (ssmap_it_type iter = m_params.begin(); iter != m_params.end(); ++iter)
        std::cout << "   " << iter->first << ": " << iter->second << std::endl;
}

bool configurable::has_param(const std::string &key) const {
    return static_cast<bool>(m_params.count(key));
}

} // namespace libresponse
