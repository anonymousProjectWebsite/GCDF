#ifndef PARAMETERS_MANAGER_H
#define PARAMETERS_MANAGER_H

#include "common/BasicTypes.h"
#include <string>
#include <unordered_map>
#include <variant>


namespace CRISP {
using ParameterValue = std::variant<vector_t, CDFParameters>;
class ParametersManager {
public:

    ParametersManager() = default;
    // Set parameters associated with a specific name
    void setParameters(const std::string& name, const ParameterValue& params) {
        parameters_[name] = params;
    }

    // Retrieve parameters associated with a specific name
    template<typename T = vector_t>
    T getParameters(const std::string& name) const {
        auto it = parameters_.find(name);
        if (it != parameters_.end()) {

            return std::get<T>(it->second);
        }
        throw std::runtime_error("Parameters not found for: " + name);
    }
    
    std::unordered_map<std::string, ParameterValue> getParametersMap() const {
        return parameters_;
    }

private:
    std::unordered_map<std::string, ParameterValue> parameters_;
};
}

#endif // PARAMETERS_MANAGER_H