#ifndef BASIC_TYPES_H
#define BASIC_TYPES_H

#include <cppad/cg.hpp>
#include <cppad/cppad.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <vector>

namespace CRISP{
// Basic types
using scalar_t = double;
using SizeVector = std::vector<size_t>;
using ValueVector = std::vector<scalar_t>;

// Eigen vector and matrix types
using vector_t = Eigen::VectorXd;
using matrix_t = Eigen::MatrixXd;
using sparse_vector_t = Eigen::SparseVector<scalar_t>;
using sparse_matrix_t = Eigen::SparseMatrix<scalar_t, Eigen::RowMajor>;
// define a structure with outterIndex, innerIndices and values
struct CSRSparseMatrix {
    SizeVector outerIndex;
    SizeVector innerIndices;
    ValueVector values;
    CSRSparseMatrix() = default;
    CSRSparseMatrix(const size_t& rows, const size_t& nnz) {
        outerIndex.resize(rows + 1);
        outerIndex[0] = 0; // first element is always 0
        innerIndices.resize(nnz);
        values.resize(nnz);
    }
    CSRSparseMatrix(const SizeVector& outer, const SizeVector& inner, const ValueVector& vals) : outerIndex(outer), innerIndices(inner), values(vals) {}
    CSRSparseMatrix(const CSRSparseMatrix& other) {
        std::memcpy(outerIndex.data(), other.outerIndex.data(), other.outerIndex.size() * sizeof(int));
        std::memcpy(innerIndices.data(), other.innerIndices.data(), other.innerIndices.size() * sizeof(int));
        std::memcpy(values.data(), other.values.data(), other.values.size() * sizeof(scalar_t));
    }
    
    void toEigenSparseMatrix(sparse_matrix_t &sparseMatrix) {
        std::vector<int> outerIndex_int(outerIndex.begin(), outerIndex.end());
        std::vector<int> innerIndices_int(innerIndices.begin(), innerIndices.end());
        std::memcpy(sparseMatrix.outerIndexPtr(), outerIndex_int.data(), outerIndex_int.size() * sizeof(int));
        std::memcpy(sparseMatrix.innerIndexPtr(), innerIndices_int.data(), innerIndices_int.size() * sizeof(int));
        std::memcpy(sparseMatrix.valuePtr(), values.data(), values.size() * sizeof(scalar_t));
    }
    void print() {
        std::cout << "OuterIndex (row Pointers):" << std::endl;
        for (size_t i = 0; i < outerIndex.size(); ++i) {
            std::cout << outerIndex[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "InnerIndices (colum Indices):" << std::endl;
        for (size_t i = 0; i < innerIndices.size(); ++i) {
            std::cout << innerIndices[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Values:" << std::endl;
        for (size_t i = 0; i < values.size(); ++i) {
            std::cout << values[i] << " ";
        }
    }
};

// structure for the parameters of the CDF (configuration distance field) constraints:
// obs_lists: std::vector<vector_t> where each vector_t is a the obstacle position vector
// obs_index_mapping: std::vector<std::vector<int>> where each vector<int> is the index mapping of each time step for the obstacles, namely the obs points that need to be considered at each time step
// obs_radius: to be added
struct CDFParameters {
    std::vector<vector_t> obs_lists;
    std::vector<std::vector<int>> obs_index_mapping;
    size_t batch_size; // batch size for query the function
    size_t num_nonzeros; // number of nonzeros in the sparsity pattern
    scalar_t bias_height; // height of the mobile manipulator base, used to transform the world frame to the mobile base frame
    size_t preset_batch_size;
    size_t num_configuration_variables_per_step;
    SizeVector outerIndex;
    SizeVector innerIndices;
    CDFParameters() = default;
    CDFParameters(const std::vector<vector_t>& obs_lists, const std::vector<std::vector<int>>& obs_index_mapping, const size_t preset_batch, const size_t num_configuration_variables, const scalar_t bias_z = 0.5)
        : obs_lists(obs_lists), obs_index_mapping(obs_index_mapping){
            // batch size should be equal to the sum of the size of obs_index_mapping
            batch_size = 0;
            for (const auto& obs_index : obs_index_mapping) {
                batch_size += obs_index.size();
            }
            bias_height = bias_z;
            preset_batch_size = preset_batch;
            num_configuration_variables_per_step = num_configuration_variables;
            num_nonzeros = batch_size * num_configuration_variables_per_step; // number of nonzeros in the sparsity pattern
            updateSparsityPattern();
        }
    
    std::vector<vector_t> getObsLists() const {
        return obs_lists;
    }
    
    std::vector<std::vector<int>> getObsIndexMapping() const {
        return obs_index_mapping;
    }

    size_t getBatchSize() const {
        return batch_size;
    }

    size_t getPresetBatchSize() const {
        return preset_batch_size;
    }

    scalar_t getBiasHeight() const {
        return bias_height;
    }

    size_t getNumNonZerosJacobian() const {
        return num_nonzeros;
    }
    SizeVector getOuterIndex() const {
        return outerIndex;
    }
    SizeVector getInnerIndices() const {
        return innerIndices;
    }
    
    size_t getNumConfigurationVariablesPerStep() const {
        return num_configuration_variables_per_step;
    }   

    void changeObsAssign(const std::vector<vector_t>& new_obs_lists, const std::vector<std::vector<int>>& new_obs_index_mapping) {
        obs_lists = new_obs_lists;
        obs_index_mapping = new_obs_index_mapping;
        // update batch size and sparsity pattern
        batch_size = 0;
        for (const auto& obs_index : obs_index_mapping) {
            batch_size += obs_index.size();
        }
        updateSparsityPattern();
    }

    void updateSparsityPattern() {
        outerIndex.resize(batch_size + 1);
        outerIndex[0] = 0; // first element is always 0
        innerIndices.resize(num_configuration_variables_per_step * batch_size);
        size_t current_row = 0;
        size_t num_nonzeros = 0;
        for (size_t i = 0; i < obs_index_mapping.size(); ++i) {
            SizeVector current_column_indices(num_configuration_variables_per_step); //innerindex is the same for each time step, each time step has 2 * num_configuration_variables_per_step variables.
            for (size_t j = 0; j < num_configuration_variables_per_step; ++j) {
                current_column_indices[j] = j + i * (num_configuration_variables_per_step * 2);
            }
            for (const auto& index : obs_index_mapping[i]) {
                if (index < obs_lists.size()) {
                    // for each obs point, we need to add the corresponding row and column index
                    outerIndex[current_row + 1] = outerIndex[current_row] + num_configuration_variables_per_step;
                    std::memcpy(innerIndices.data() + num_nonzeros, current_column_indices.data(), current_column_indices.size() * sizeof(size_t));
                    current_row++;
                    num_nonzeros += num_configuration_variables_per_step;
                } else {
                    throw std::runtime_error("Index out of bounds in obs_index_mapping.");
                }
            }
        } 
    }
};


// AD vector types (can use std::vector or Eigen containers)
using cg_scalar_t = CppAD::cg::CG<scalar_t>;
using ad_scalar_t = CppAD::AD<cg_scalar_t>;
using ad_vector_std = std::vector<ad_scalar_t>;
using ad_vector_t = Eigen::Matrix<ad_scalar_t, Eigen::Dynamic, 1>;
using ad_matrix_t = Eigen::Matrix<ad_scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
// AD function types
using ad_function_t = std::function<void(const ad_vector_t&, ad_vector_t&)>;
using ad_function_with_param_t = std::function<void(const ad_vector_t&, const ad_vector_t&, ad_vector_t&)>;
using triplet_vector_t = std::vector<Eigen::Triplet<scalar_t>>;
}   
#endif // BASIC_TYPES_H
