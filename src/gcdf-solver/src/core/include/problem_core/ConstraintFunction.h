#ifndef CONSTRAINT_FUNCTION_H
#define CONSTRAINT_FUNCTION_H

#include "common/ValueFunction.h"
// include casadi
#include <casadi/casadi.hpp>

// Only first order information is needed for constraints
namespace CRISP {
    class ConstraintFunction : public ValueFunction {
    public:
        enum class SpecifiedFunctionLevel {
            NONE,
            VALUE,
            GRADIENT
        };
        ConstraintFunction(size_t variableDim, const std::string& modelName, const std::string& folderName,
                           const std::string& functionName, const ad_function_t& function, bool regenerateLibrary = false,
                           CppAdInterface::ModelInfoLevel infoLevel = CppAdInterface::ModelInfoLevel::FIRST_ORDER, SpecifiedFunctionLevel specifiedFunctionLevel = SpecifiedFunctionLevel::NONE):
                           specifiedFunctionLevel_(specifiedFunctionLevel), functionName_(functionName), isParameterized_(false){
            cppadInterface_ = std::make_unique<CppAdInterface>(variableDim, modelName, folderName, functionName, function, infoLevel, regenerateLibrary);
            nnzJacobian_ = cppadInterface_->getNumNonZerosJacobian();
            variableDim_ = variableDim;
            funDim_ = cppadInterface_->getFunDim();        
        }

        ConstraintFunction(size_t variableDim, size_t parameterDim, const std::string& modelName, const std::string& folderName,
                           const std::string& functionName, const ad_function_with_param_t& function,  bool regenerateLibrary = false,
                           CppAdInterface::ModelInfoLevel infoLevel = CppAdInterface::ModelInfoLevel::FIRST_ORDER, SpecifiedFunctionLevel specifiedFunctionLevel = SpecifiedFunctionLevel::NONE): 
                           specifiedFunctionLevel_(specifiedFunctionLevel), functionName_(functionName), isParameterized_(true){
            cppadInterface_ = std::make_unique<CppAdInterface>(variableDim, parameterDim, modelName, folderName, functionName, function, infoLevel, regenerateLibrary);
            nnzJacobian_ = cppadInterface_->getNumNonZerosJacobian();
            variableDim_ = variableDim;
            parameterDim_ = parameterDim;
            funDim_ = cppadInterface_->getFunDim();
        }


        // for cdf constraints, value function and gradient function are casadi functions
        ConstraintFunction(size_t variableDim, const casadi::Function& cdfValueFunction, const casadi::Function& cdfGradientFunction, CDFParameters& cdfParameters,
                           const std::string& functionName, SpecifiedFunctionLevel specifiedFunctionLevel = SpecifiedFunctionLevel::GRADIENT):
                           specifiedFunctionLevel_(specifiedFunctionLevel), functionName_(functionName), isParameterized_(false), externalCDFValueFunction_(cdfValueFunction), externalCDFGradientFunction_(cdfGradientFunction) {
            // // sdf is used to determine the sign of cdf and its gradients
            // variableDim_ = variableDim; // total variables dimension
            // funDim_ = cdfParameters.getBatchSize(); // be careful that this would change according to the parameters, equals to batch size
            // numConfigurationVariablesPerStep_ = cdfParameters.getNumConfigurationVariablesPerStep(); // number of configuration variables per step
            // // check if the cdfValueFunction has the correct input dimension
            // if (numConfigurationVariablesPerStep_ + 3 != externalCDFValueFunction_.size_in(0).first) {
            //     throw std::runtime_error("Number of configuration variables per step does not match the input dimension of the CDF value function.");
            // }
            // // we have a known block diagnal sparsity pattern for the cdf constraints
            // // get cdf constraints variables, need to be updated when set new CDF parameters
            // nnzJacobian_ = funDim_ * numConfigurationVariablesPerStep_; // equals to the size of inner indices of the sparsity pattern.
            // obs_lists_ = cdfParameters.getObsLists();
            // obs_index_mapping_ = cdfParameters.getObsIndexMapping();
            // batch_size_ = cdfParameters.getBatchSize();
            // maximum_batch_size_ = cdfParameters.getPresetBatchSize();
            // z_ = cdfParameters.getBiasHeight(); // height of the mobile manipulator base, used to transform the world frame to the mobile base frame
            // outerIndex_cdf_ = cdfParameters.getOuterIndex();
            // innerIndices_cdf_ = cdfParameters.getInnerIndices();


            // // std::cout << "number of configuration variables per step: " << numConfigurationVariablesPerStep_ << std::endl;
            // cdfValueFunction_ = [this](const vector_t& x) -> vector_t {
            //     // 1. construct inputs eigen from x accoriding to the obs_index_mapping and obs_lists
            //     // 2. transform it into casadi DM formate and query the casadi value function
            //     // 3. transform the output DM to an eigen vector.
            //     // x is the optimization variable, p would be the cdf parameter structure
            //     if (batch_size_ > maximum_batch_size_) {
            //         throw std::runtime_error("Batch size exceeds the preset maximum batch size."); // padding zeros for those greater than batch_size_
            //     }
            //     // input dimension is （numConfigurationVariablesPerStep_ + 3） * batch size
            //     vector_t input_eigen((numConfigurationVariablesPerStep_ + 3) * maximum_batch_size_);
            //     input_eigen.setZero(); // initialize to zero
            //     size_t current_idx = 0;
            //     for (size_t i = 0; i < obs_index_mapping_.size(); ++i) { // obs_index_mapping size should equals to the number of time steps - 1, cause for the initial time step, we deem it safe
            //         // for each time step, we need to extract the corresponding obs points
            //         if (obs_index_mapping_[i].empty()){
            //             continue; // skip if no obs points for this time step
            //         }
            //         for (int index : obs_index_mapping_[i]) {
            //             if (index < obs_lists_.size()) {
            //                 // if not empty, constuct the input vector of this sample point
            //                 vector_t input_current(numConfigurationVariablesPerStep_ + 3);
            //                 input_current.segment(0, 2) = obs_lists_[index].segment(0, 2) - x.segment((i+1) * (2 * numConfigurationVariablesPerStep_), 2);
            //                 input_current(2) = obs_lists_[index](2); // biased height
            //                 input_current(3) = 0.0; input_current(4) = 0.0; // query this value from robot base frame.
            //                 input_current.segment(5, numConfigurationVariablesPerStep_ - 2) = x.segment((i+1) * (2 * numConfigurationVariablesPerStep_) + 2, numConfigurationVariablesPerStep_ - 2);
                            
            //                 // use memory copy to add input current into the total
            //                 std::memcpy(input_eigen.data() + current_idx, input_current.data(), sizeof(scalar_t) * input_current.size()); // only extract actual batch_size_ values, leaving remaining as zero
            //                 current_idx += input_current.size();
            //             } else {
            //                 throw std::runtime_error("Index out of bounds in obs_index_mapping.");
            //             }
            //         }
            //     }
            //     casadi::DM input_dm = casadi::DM::reshape(casadi::DM(std::vector<scalar_t>(input_eigen.data(), input_eigen.data() + input_eigen.size())), numConfigurationVariablesPerStep_ + 3, maximum_batch_size_);
            //     std::vector<casadi::DM> result_cdf = externalCDFValueFunction_(input_dm); // by test, result of casadi function evaluation should be a std::vector if it is not a scalar 
            //     // casadi::DM result_sdf = externalSDFValueFunction_(inputs);
            //     // use sdf value to determine the sign of cdf: if sdf is less than 0, cdf sign should be reversed.
            //     // std::vector<casadi::DM> result_sdf = externalSDFValueFunction_(input_dm);
            //     // process result to output an eigen vector_t
            //     vector_t output_cdf_eigen(batch_size_); // assume result_cdf is a column major data
            //     // vector_t output_sdf_eigen(batch_size_);
            //     // ValueVector result_sdf_values(result_sdf[0].nonzeros()); 
            //     std::memcpy(output_cdf_eigen.data(), result_cdf[0].ptr(), sizeof(scalar_t) * batch_size_); // only extract actual batch_size_ constraints
            //     // std::memcpy(output_sdf_eigen.data(), result_sdf[0].ptr(), sizeof(scalar_t) * batch_size_); // only extract actual batch_size_ constraints
            //     // if sdf is less than 0, reverse the sign of cdf
            //     // for (size_t i = 0; i < output_cdf_eigen.size(); ++i) {
            //     //     if (result_sdf_values[i] < 0) {
            //     //         output_cdf_eigen(i) = -output_cdf_eigen(i);
            //     //     }
            //     // }
            //     // - threshold for cdf value to avoid numerical issues
            //     vector_t threshold_vector = vector_t::Constant(output_cdf_eigen.size(), cdf_threshold_);
            //     output_cdf_eigen = output_cdf_eigen - threshold_vector;
            //     return output_cdf_eigen;
            // };

            // cdfGradientFunction_ = [this](const vector_t& x) -> CSRSparseMatrix {
            //     // 1. construct inputs eigen from x accoriding to the obs_index_mapping and obs_lists
            //     // 2. transform it into casadi DM formate and query the casadi gradient function
            //     // 3. cast the dense gradient vector to eigen sparse matrix using our CSRSparseMatrix
            //     // step 3 is quiet tricky, since we need to utilize the sparsity pattern of the CDF constraints.
            //     // be careful of: (i) dimension difference between cdf function and our configuration variables. 
            //                 //    (ii) to maximize the computational efficiency, user are required to preprocess the cdf gradient functions when creating that, so I can directly use the nonzero values here.
            //     // construct the std::vector<casadi::DM> inputs from x accoriding to the obs_index_mapping and obs_lists:
            //     vector_t input_eigen((numConfigurationVariablesPerStep_ + 3) * maximum_batch_size_);
            //     input_eigen.setZero(); // initialize to zero
            //     size_t current_idx = 0;
            //     // CSR construction variable
            //     // CSRSparseMatrix output_cdf_gradient_csr(funDim_, nnzJacobian_);
            //     // sparsity pattern of cdf constraints is determined according to the parameters
            //     for (size_t i = 0; i < obs_index_mapping_.size(); ++i) { // obs_index_mapping size should equals to the number of time steps - 1 
            //         // for each time step, we need to extract the corresponding obs points
            //         if (obs_index_mapping_[i].empty()){
            //             continue; // skip if no obs points for this time step
            //         }
            //         for (int index : obs_index_mapping_[i]) {
            //             if (index < obs_lists_.size()) {
            //                 // if not empty, constuct the input vector of this sample point
            //                 vector_t input_current(numConfigurationVariablesPerStep_ + 3);
            //                 input_current.segment(0, 2) = obs_lists_[index].segment(0, 2) - x.segment((i+1) * (2 * numConfigurationVariablesPerStep_), 2);
            //                 input_current(2) = obs_lists_[index](2); // biased height
            //                 input_current(3) = 0.0; input_current(4) = 0.0; // query this value from robot base frame.
            //                 input_current.segment(5, numConfigurationVariablesPerStep_ - 2) = x.segment((i+1) * (2 * numConfigurationVariablesPerStep_) + 2, numConfigurationVariablesPerStep_ - 2);
                            
            //                 // use memory copy to add input current into the total
            //                 std::memcpy(input_eigen.data() + current_idx, input_current.data(), sizeof(scalar_t) * input_current.size()); // only extract actual batch_size_ values, leaving remaining as zero
            //                 current_idx += input_current.size();
            //             } else {
            //                 throw std::runtime_error("Index out of bounds in obs_index_mapping.");
            //             }
            //         }
            //     }
            //     casadi::DM input_dm = casadi::DM::reshape(casadi::DM(std::vector<scalar_t>(input_eigen.data(), input_eigen.data() + input_eigen.size())), numConfigurationVariablesPerStep_ + 3, maximum_batch_size_);
            //     std::vector<casadi::DM> result_cdf_gradient = externalCDFGradientFunction_(input_dm); // by test, result of casadi function evaluation should be a std::vector if it is not a scalar 
            //     // std::vector<casadi::DM> result_sdf = externalSDFValueFunction_(input_dm); // sdf is used to determine the sign of cdf and its gradients
            //     // ValueVector result_sdf_values(result_sdf[0].nonzeros());
            //     ValueVector nz(result_cdf_gradient[0].nonzeros()); // !!assume you have preprocessed the cdf gradient function when creating the function handle accounting for the chain rule and dimension alignment.
            //     ValueVector nz_truncated(nz.begin(), nz.begin() + batch_size_ * numConfigurationVariablesPerStep_); // only extract actual batch_size_ constraints
            //     // reverse the sign of gradient if sdf of that constraint is less than 0
            //     // for (size_t i = 0; i < batch_size_; ++i) {
            //     //     if (result_sdf_values[i] < 0) {
            //     //         for (size_t j = 0; j < numConfigurationVariablesPerStep_; ++j) {
            //     //             nz_truncated[i * numConfigurationVariablesPerStep_ + j] = -nz_truncated[i * numConfigurationVariablesPerStep_ + j];
            //     //         }
            //     //     }
            //     // }
            //     CSRSparseMatrix output_cdf_gradient_csr(outerIndex_cdf_, innerIndices_cdf_, nz_truncated);
            //     return output_cdf_gradient_csr; // return the CSR sparse matrix
            // };
        }

        ConstraintFunction(size_t variableDim, const std::string& cdfValueFunctionStr, const std::string& cdfGradientFunctionStr, CDFParameters& cdfParameters,
            const std::string& functionName, SpecifiedFunctionLevel specifiedFunctionLevel = SpecifiedFunctionLevel::GRADIENT):
            specifiedFunctionLevel_(specifiedFunctionLevel), functionName_(functionName), isParameterized_(false) {
        // sdf is used to determine the sign of cdf and its gradients
        variableDim_ = variableDim; // total variables dimension
        externalCDFValueFunction_ = casadi::Function::deserialize(cdfValueFunctionStr);
        externalCDFGradientFunction_ = casadi::Function::deserialize(cdfGradientFunctionStr);
        funDim_ = cdfParameters.getBatchSize(); // be careful that this would change according to the parameters, equals to batch size
        numConfigurationVariablesPerStep_ = cdfParameters.getNumConfigurationVariablesPerStep(); // number of configuration variables per step
        // check if the cdfValueFunction has the correct input dimension
        if (numConfigurationVariablesPerStep_ + 3 != externalCDFValueFunction_.size_in(0).first) {
        throw std::runtime_error("Number of configuration variables per step does not match the input dimension of the CDF value function.");
        }
        // we have a known block diagnal sparsity pattern for the cdf constraints
        // get cdf constraints variables, need to be updated when set new CDF parameters
        nnzJacobian_ = funDim_ * numConfigurationVariablesPerStep_; // equals to the size of inner indices of the sparsity pattern.
        obs_lists_ = cdfParameters.getObsLists();
        obs_index_mapping_ = cdfParameters.getObsIndexMapping();
        batch_size_ = cdfParameters.getBatchSize();
        maximum_batch_size_ = cdfParameters.getPresetBatchSize();
        z_ = cdfParameters.getBiasHeight(); // height of the mobile manipulator base, used to transform the world frame to the mobile base frame
        outerIndex_cdf_ = cdfParameters.getOuterIndex();
        innerIndices_cdf_ = cdfParameters.getInnerIndices();


        // std::cout << "number of configuration variables per step: " << numConfigurationVariablesPerStep_ << std::endl;
        cdfValueFunction_ = [this](const vector_t& x) -> vector_t {
        // 1. construct inputs eigen from x accoriding to the obs_index_mapping and obs_lists
        // 2. transform it into casadi DM formate and query the casadi value function
        // 3. transform the output DM to an eigen vector.
        // x is the optimization variable, p would be the cdf parameter structure
        if (batch_size_ > maximum_batch_size_) {
            throw std::runtime_error("Batch size exceeds the preset maximum batch size."); // padding zeros for those greater than batch_size_
        }
        // input dimension is （numConfigurationVariablesPerStep_ + 3） * batch size
        vector_t input_eigen((numConfigurationVariablesPerStep_ + 3) * maximum_batch_size_);
        input_eigen.setZero(); // initialize to zero
        size_t current_idx = 0;
        for (size_t i = 0; i < obs_index_mapping_.size(); ++i) { // obs_index_mapping size should equals to the number of time steps - 1, cause for the initial time step, we deem it safe
            // for each time step, we need to extract the corresponding obs points
            if (obs_index_mapping_[i].empty()){
                continue; // skip if no obs points for this time step
            }
            for (int index : obs_index_mapping_[i]) {
                if (index < obs_lists_.size()) {
                    // if not empty, constuct the input vector of this sample point
                    vector_t input_current(numConfigurationVariablesPerStep_ + 3);
                    input_current.segment(0, 2) = obs_lists_[index].segment(0, 2) - x.segment((i+1) * (2 * numConfigurationVariablesPerStep_), 2);
                    input_current(2) = obs_lists_[index](2); // biased height
                    input_current(3) = 0.0; input_current(4) = 0.0; // query this value from robot base frame.
                    input_current.segment(5, numConfigurationVariablesPerStep_ - 2) = x.segment((i+1) * (2 * numConfigurationVariablesPerStep_) + 2, numConfigurationVariablesPerStep_ - 2);
                    
                    // use memory copy to add input current into the total
                    std::memcpy(input_eigen.data() + current_idx, input_current.data(), sizeof(scalar_t) * input_current.size()); // only extract actual batch_size_ values, leaving remaining as zero
                    current_idx += input_current.size();
                } else {
                    throw std::runtime_error("Index out of bounds in obs_index_mapping.");
                }
            }
        }
        casadi::DM input_dm = casadi::DM::reshape(casadi::DM(std::vector<scalar_t>(input_eigen.data(), input_eigen.data() + input_eigen.size())), numConfigurationVariablesPerStep_ + 3, maximum_batch_size_);
        std::vector<casadi::DM> result_cdf = externalCDFValueFunction_(input_dm); // by test, result of casadi function evaluation should be a std::vector if it is not a scalar 
        // casadi::DM result_sdf = externalSDFValueFunction_(inputs);
        // use sdf value to determine the sign of cdf: if sdf is less than 0, cdf sign should be reversed.
        // std::vector<casadi::DM> result_sdf = externalSDFValueFunction_(input_dm);
        // process result to output an eigen vector_t
        vector_t output_cdf_eigen(batch_size_); // assume result_cdf is a column major data
        // vector_t output_sdf_eigen(batch_size_);
        // ValueVector result_sdf_values(result_sdf[0].nonzeros()); 
        std::memcpy(output_cdf_eigen.data(), result_cdf[0].ptr(), sizeof(scalar_t) * batch_size_); // only extract actual batch_size_ constraints
        // std::memcpy(output_sdf_eigen.data(), result_sdf[0].ptr(), sizeof(scalar_t) * batch_size_); // only extract actual batch_size_ constraints
        // if sdf is less than 0, reverse the sign of cdf
        // for (size_t i = 0; i < output_cdf_eigen.size(); ++i) {
        //     if (result_sdf_values[i] < 0) {
        //         output_cdf_eigen(i) = -output_cdf_eigen(i);
        //     }
        // }
        // - threshold for cdf value to avoid numerical issues
        vector_t bias_vector = vector_t::Constant(output_cdf_eigen.size(), cdf_bias_);
        output_cdf_eigen = output_cdf_eigen - bias_vector;
        return output_cdf_eigen;
        };

        cdfGradientFunction_ = [this](const vector_t& x) -> CSRSparseMatrix {
        // 1. construct inputs eigen from x accoriding to the obs_index_mapping and obs_lists
        // 2. transform it into casadi DM formate and query the casadi gradient function
        // 3. cast the dense gradient vector to eigen sparse matrix using our CSRSparseMatrix
        // step 3 is quiet tricky, since we need to utilize the sparsity pattern of the CDF constraints.
        // be careful of: (i) dimension difference between cdf function and our configuration variables. 
                    //    (ii) to maximize the computational efficiency, user are required to preprocess the cdf gradient functions when creating that, so I can directly use the nonzero values here.
        // construct the std::vector<casadi::DM> inputs from x accoriding to the obs_index_mapping and obs_lists:
        vector_t input_eigen((numConfigurationVariablesPerStep_ + 3) * maximum_batch_size_);
        input_eigen.setZero(); // initialize to zero
        size_t current_idx = 0;
        // CSR construction variable
        // CSRSparseMatrix output_cdf_gradient_csr(funDim_, nnzJacobian_);
        // sparsity pattern of cdf constraints is determined according to the parameters
        for (size_t i = 0; i < obs_index_mapping_.size(); ++i) { // obs_index_mapping size should equals to the number of time steps - 1 
            // for each time step, we need to extract the corresponding obs points
            if (obs_index_mapping_[i].empty()){
                continue; // skip if no obs points for this time step
            }
            for (int index : obs_index_mapping_[i]) {
                if (index < obs_lists_.size()) {
                    // if not empty, constuct the input vector of this sample point
                    vector_t input_current(numConfigurationVariablesPerStep_ + 3);
                    input_current.segment(0, 2) = obs_lists_[index].segment(0, 2) - x.segment((i+1) * (2 * numConfigurationVariablesPerStep_), 2);
                    input_current(2) = obs_lists_[index](2); // biased height
                    input_current(3) = 0.0; input_current(4) = 0.0; // query this value from robot base frame.
                    input_current.segment(5, numConfigurationVariablesPerStep_ - 2) = x.segment((i+1) * (2 * numConfigurationVariablesPerStep_) + 2, numConfigurationVariablesPerStep_ - 2);
                    
                    // use memory copy to add input current into the total
                    std::memcpy(input_eigen.data() + current_idx, input_current.data(), sizeof(scalar_t) * input_current.size()); // only extract actual batch_size_ values, leaving remaining as zero
                    current_idx += input_current.size();
                } else {
                    throw std::runtime_error("Index out of bounds in obs_index_mapping.");
                }
            }
        }
        casadi::DM input_dm = casadi::DM::reshape(casadi::DM(std::vector<scalar_t>(input_eigen.data(), input_eigen.data() + input_eigen.size())), numConfigurationVariablesPerStep_ + 3, maximum_batch_size_);
        std::vector<casadi::DM> result_cdf_gradient = externalCDFGradientFunction_(input_dm); // by test, result of casadi function evaluation should be a std::vector if it is not a scalar 
        // std::vector<casadi::DM> result_sdf = externalSDFValueFunction_(input_dm); // sdf is used to determine the sign of cdf and its gradients
        // ValueVector result_sdf_values(result_sdf[0].nonzeros());
        ValueVector nz(result_cdf_gradient[0].nonzeros()); // !!assume you have preprocessed the cdf gradient function when creating the function handle accounting for the chain rule and dimension alignment.
        ValueVector nz_truncated(nz.begin(), nz.begin() + batch_size_ * numConfigurationVariablesPerStep_); // only extract actual batch_size_ constraints
        // reverse the sign of gradient if sdf of that constraint is less than 0
        // for (size_t i = 0; i < batch_size_; ++i) {
        //     if (result_sdf_values[i] < 0) {
        //         for (size_t j = 0; j < numConfigurationVariablesPerStep_; ++j) {
        //             nz_truncated[i * numConfigurationVariablesPerStep_ + j] = -nz_truncated[i * numConfigurationVariablesPerStep_ + j];
        //         }
        //     }
        // }
        CSRSparseMatrix output_cdf_gradient_csr(outerIndex_cdf_, innerIndices_cdf_, nz_truncated);
        return output_cdf_gradient_csr; // return the CSR sparse matrix
        };
        }



        //  for pybind
        ConstraintFunction(size_t variableDim, const std::string& modelName, const std::string& folderName,
                           const std::string& functionName, bool regenerateLibrary = false,
                           CppAdInterface::ModelInfoLevel infoLevel = CppAdInterface::ModelInfoLevel::FIRST_ORDER, SpecifiedFunctionLevel specifiedFunctionLevel = SpecifiedFunctionLevel::NONE):
                           specifiedFunctionLevel_(specifiedFunctionLevel), functionName_(functionName), isParameterized_(false){

            cppadInterface_ = std::make_unique<CppAdInterface>(variableDim, modelName, folderName, functionName, infoLevel, regenerateLibrary);
            nnzJacobian_ = cppadInterface_->getNumNonZerosJacobian();
            variableDim_ = variableDim;
            funDim_ = cppadInterface_->getFunDim();
        }

        ConstraintFunction(size_t variableDim, size_t parameterDim, const std::string& modelName, const std::string& folderName,
                           const std::string& functionName, bool regenerateLibrary = false,
                           CppAdInterface::ModelInfoLevel infoLevel = CppAdInterface::ModelInfoLevel::FIRST_ORDER, SpecifiedFunctionLevel specifiedFunctionLevel = SpecifiedFunctionLevel::NONE): 
                           specifiedFunctionLevel_(specifiedFunctionLevel), functionName_(functionName), isParameterized_(true){
            //   convert the eigen vector function to ad function
            cppadInterface_ = std::make_unique<CppAdInterface>(variableDim, parameterDim, modelName, folderName, functionName, infoLevel, regenerateLibrary);
            nnzJacobian_ = cppadInterface_->getNumNonZerosJacobian();
            variableDim_ = variableDim;
            parameterDim_ = parameterDim;
            funDim_ = cppadInterface_->getFunDim();
        }


        //  ------------------------ Get function information from ad or user defined functions ------------------------ //
        vector_t getValue(const vector_t& x, const vector_t& params) override {
            if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::VALUE) {
                if (valueFunctionWithParam_ != nullptr) {
                    return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.") : valueFunctionWithParam_(x, params);
                } else {
                    throw std::runtime_error("No value function with parameters specified.");
                }
            }
            return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.") : cppadInterface_->computeFunctionValue(x, params);
        }

        vector_t getValue(const vector_t& x) override {
            if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::VALUE) {
                if (valueFunction_ != nullptr) {
                    return isParameterized_ ? throw std::runtime_error("Parameters are required.") : valueFunction_(x);
                } else if (cdfValueFunction_ != nullptr) {
                    // if cdf value function is specified, use it to compute the value
                    return isParameterized_ ? throw std::runtime_error("Parameters are required.") : cdfValueFunction_(x);
                } else {
                    throw std::runtime_error("No value function specified.");
                }
            }
            return isParameterized_ ? throw std::runtime_error("Parameters are required.") : cppadInterface_->computeFunctionValue(x);
        }

        triplet_vector_t getGradientTriplet(const vector_t& x, const vector_t& params) {
            if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::GRADIENT) {
                if (gradientFunctionWithParam_ != nullptr) {
                    return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.") : cppadInterface_->computeSparseJacobianTriplet(x, params);
                } else {
                    throw std::runtime_error("No gradient function with parameters specified.");
                }
            }
            return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.") : cppadInterface_->computeSparseJacobianTriplet(x, params);
        }

        triplet_vector_t getGradientTriplet(const vector_t& x) {
            if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::GRADIENT) {
                if (gradientFunction_ != nullptr) {
                    return isParameterized_ ? throw std::runtime_error("Parameters are required.") : cppadInterface_->computeSparseJacobianTriplet(x);
                } else {
                    throw std::runtime_error("No gradient function specified.");
                }
            }
            return isParameterized_ ? throw std::runtime_error("Parameters are required.") : cppadInterface_->computeSparseJacobianTriplet(x);
        }

        sparse_matrix_t getGradient(const vector_t& x, const vector_t& params) override {
            if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::GRADIENT) {
                if (gradientFunctionWithParam_ != nullptr) {
                    return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.") : gradientFunctionWithParam_(x, params);
                } else {
                    throw std::runtime_error("No gradient function with parameters specified.");
                }
            }
            return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.") : cppadInterface_->computeSparseJacobian(x, params);
        }

        sparse_matrix_t getGradient(const vector_t& x) override {
            if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::GRADIENT) {
                if (gradientFunction_ != nullptr) {
                    return isParameterized_ ? throw std::runtime_error("Parameters are required.") : gradientFunction_(x);
                } else {
                    throw std::runtime_error("No gradient function specified.");
                }
            }
            return isParameterized_ ? throw std::runtime_error("Parameters are required.") : cppadInterface_->computeSparseJacobian(x);
        }

        CSRSparseMatrix getGradientCSR(const vector_t& x, const vector_t& params) {
            if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::GRADIENT) {
                if (gradientFunctionWithParam_ != nullptr) {
                    return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.") : cppadInterface_->computeSparseJacobianCSR(x, params);
                } else {
                    throw std::runtime_error("No gradient function with parameters specified.");
                }
            }
            return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.") : cppadInterface_->computeSparseJacobianCSR(x, params);
        }

        CSRSparseMatrix getGradientCSR(const vector_t& x) {
            if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::GRADIENT) {
                if (gradientFunction_ != nullptr) {
                    return isParameterized_ ? throw std::runtime_error("Parameters are not expected.") : cppadInterface_->computeSparseJacobianCSR(x);
                } else if (cdfGradientFunction_ != nullptr) {
                    // if cdf gradient function is specified, use it to compute the sparse jacobian
                    return isParameterized_ ? throw std::runtime_error("Parameters are not expected.") : cdfGradientFunction_(x);
                } else {
                    throw std::runtime_error("No gradient function specified.");
                }
            }
            return isParameterized_ ? throw std::runtime_error("Parameters are required.") : cppadInterface_->computeSparseJacobianCSR(x);
        }

        SpecifiedFunctionLevel getSpecifiedFunctionLevel() const {
            return specifiedFunctionLevel_;
        }

        bool isParameterized() const {
            return isParameterized_;
        }

        size_t getVariableDim() const {
            return variableDim_;
        }

        size_t getFunDim() const {
            return funDim_;
        }

        size_t getParameterDim() const {
            return parameterDim_;
        }

        size_t getNumNonZerosJacobian() const {
            return nnzJacobian_;
        }

        void setFunDim(size_t funDim) {
            funDim_ = funDim;
        }

        void setNumNonZerosJacobian(size_t nnzJacobian) {
            nnzJacobian_ = nnzJacobian;
        }

        void setCDFparameters(const CDFParameters& cdfParameters) {
            // set the cdf parameters, including obs_lists_, obs_index_mapping_, batch_size_, maximum_batch_size_, z_, outerIndex_cdf_, innerIndices_cdf_
            numConfigurationVariablesPerStep_ = cdfParameters.getNumConfigurationVariablesPerStep();
            if (numConfigurationVariablesPerStep_ + 3 != externalCDFValueFunction_.size_in(0).first) {
                throw std::runtime_error("Number of configuration variables per step does not match the input dimension of the CDF value function.");
            }
            obs_lists_ = cdfParameters.getObsLists();
            obs_index_mapping_ = cdfParameters.getObsIndexMapping();
            batch_size_ = cdfParameters.getBatchSize();
            maximum_batch_size_ = cdfParameters.getPresetBatchSize();
            z_ = cdfParameters.getBiasHeight(); // height of the mobile manipulator base, used to transform the world frame to the mobile base frame
            outerIndex_cdf_ = cdfParameters.getOuterIndex();
            innerIndices_cdf_ = cdfParameters.getInnerIndices();
        }


        const std::string& getFunctionName() const {
            return functionName_;
        }


private:
    SpecifiedFunctionLevel specifiedFunctionLevel_;
    size_t variableDim_ = 0;
    size_t parameterDim_ = 0;
    size_t funDim_ = 0;
    size_t nnzJacobian_ = 0;
    size_t numConfigurationVariablesPerStep_ = 0; // for cdf constraints
    std::string functionName_;

    // for cdf constraints
    std::vector<vector_t> obs_lists_;
    std::vector<std::vector<int>> obs_index_mapping_;
    size_t batch_size_;
    size_t maximum_batch_size_;
    scalar_t z_ = 0;
    // scalar_t cdf_threshold_ = 2.0;
    SizeVector outerIndex_cdf_;
    SizeVector innerIndices_cdf_;
    scalar_t cdf_bias_ = 0.60; // bias for cdf value to avoid numerical issues
 
    bool isParameterized_ = false;
    // cdf and sdf casadi functions:
    casadi::Function externalCDFValueFunction_;
    // const casadi::Function externalSDFValueFunction_;
    casadi::Function externalCDFGradientFunction_;
};

} // namespace CRISP

# endif // CONSTRAINT_FUNCTION_H

