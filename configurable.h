#ifndef LIBRESPONSE_CONFIGURABLE_H_
#define LIBRESPONSE_CONFIGURABLE_H_

/*!
 * @file
 *
 * Inheritable configuration by wrapping a string->string map.
 */

#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <stdexcept>

typedef std::map< std::string, std::string > ssmap;
typedef ssmap::const_iterator ssmap_it_type;

namespace libresponse {

/*!
 * Inheritable configuration by wrapping a string->string map.
 */
class configurable {
private:
    ssmap m_params; //!< Key-value parameters
    std::string m_empty; //!< Empty value

public:
    /*!
     * Default constructor
     */
    configurable() { }

    const std::string &get_param(const std::string &key) const;
    void cfg(const std::string &key, const std::string &val);

    template<typename T>
    void get_param(const std::string &key, T &val) const;

    template<typename T>
    T get_param(const std::string &key) const;

    template<typename T>
    void cfg(const std::string &key, const T &val);

    /*!
     * Print all key-value pairs to stdout
     */
    void dump_params() const;

    bool has_param(const std::string &key) const;
};


template<typename T>
void configurable::cfg(const std::string &key, const T &val) {

    std::ostringstream ss;
    ss << val;
    cfg(key, ss.str());
}


template<typename T>
void configurable::get_param(const std::string &key, T &val) const {

    const std::string &par = get_param(key);
    if (!par.empty()) {
        std::stringstream ss;
        ss << par; ss >> val;
    }
}


template<typename T>
T configurable::get_param(const std::string &key) const {

    if(m_params.count(key) == 0) {
        throw std::logic_error(
            "configurable::get_param(): no parameter with given name: " + key);
    }

    T t;
    get_param(key, t);
    return t;
}

} // namespace libresponse

#endif // LIBRESPONSE_CONFIGURABLE_H_
