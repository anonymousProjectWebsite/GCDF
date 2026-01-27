#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include "solver_core/SolverInterface.h"
#include "common/BasicTypes.h"
#include <casadi/casadi.hpp>
// #include "common/MatlabHelper.h"

namespace py = pybind11;
using namespace CRISP;

// define the module
PYBIND11_MODULE(pyCRISP, m) {
    m.doc() = "python interface for CRISP"; // optional module docstring
    py::enum_<CppAdInterface::ModelInfoLevel>(m, "ModelInfoLevel")
        .value("FIRST_ORDER", CppAdInterface::ModelInfoLevel::FIRST_ORDER)
        .value("SECOND_ORDER", CppAdInterface::ModelInfoLevel::SECOND_ORDER)
        .export_values();

    // Register the ObjectiveFunction SpecifiedFunctionLevel enum with a unique name
    py::enum_<ObjectiveFunction::SpecifiedFunctionLevel>(m, "ObjectiveFunction_SpecifiedFunctionLevel")
        .value("NONE", ObjectiveFunction::SpecifiedFunctionLevel::NONE)
        .value("VALUE", ObjectiveFunction::SpecifiedFunctionLevel::VALUE)
        .value("GRADIENT", ObjectiveFunction::SpecifiedFunctionLevel::GRADIENT)
        .export_values();

    // Register the ConstraintFunction SpecifiedFunctionLevel enum with a unique name
    py::enum_<ConstraintFunction::SpecifiedFunctionLevel>(m, "ConstraintFunction_SpecifiedFunctionLevel")
        .value("NONE", ConstraintFunction::SpecifiedFunctionLevel::NONE)
        .value("VALUE", ConstraintFunction::SpecifiedFunctionLevel::VALUE)
        .value("GRADIENT", ConstraintFunction::SpecifiedFunctionLevel::GRADIENT)
        .export_values();
    // expose the solver interface


    py::class_<SolverInterface>(m, "SolverInterface")
        .def(py::init<OptimizationProblem&, SolverParameters&>())
        .def("initialize", &SolverInterface::initialize)
        .def("reset_problem", &SolverInterface::resetProblem) // reset problem with new initial guess
        .def("set_problem_parameters",
            static_cast<void (SolverInterface::*)(const std::string&, const vector_t&)>(&SolverInterface::setProblemParameters))
        .def("set_problem_parameters",
            static_cast<void (SolverInterface::*)(const std::string&, const CDFParameters&)>(&SolverInterface::setProblemParameters)) // problem related data, related to your obj, constraints, like the tracking reference, terminal states, etc
        .def("set_hyper_parameters", &SolverInterface::setHyperParameters) // hyperparameters for the solver, like max iterations, trust region radius, etc
        .def("solve", &SolverInterface::solve)
        .def("get_solution", &SolverInterface::getSolution)
        .def("isSuccessful", &SolverInterface::isSuccessful);
        // .def("save_results", &SolverInterface::saveResults);

    // expose optimization problem
    py::class_<OptimizationProblem>(m, "OptimizationProblem")
        .def(py::init<size_t, const std::string&>())
        .def("add_objective", &OptimizationProblem::addObjective)
        .def("add_equality_constraint", &OptimizationProblem::addEqualityConstraint)
        .def("add_inequality_constraint", &OptimizationProblem::addInequalityConstraint)
        .def("set_parameters",
            static_cast<void (OptimizationProblem::*)(const std::string&, const vector_t&)>(&OptimizationProblem::setParameters))
        .def("set_parameters",
            static_cast<void (OptimizationProblem::*)(const std::string&, const CDFParameters&)>(&OptimizationProblem::setParameters));
    
    // expose solver parameters
    py::class_<SolverParameters>(m, "SolverParameters")
        .def(py::init<>())
        .def("set_parameters", &SolverParameters::setParameters);
    
    // expose matlab helper
    // py::class_<MatlabHelper>(m, "MatlabHelper")
    //     .def_static("read_variable_from_mat_file", &MatlabHelper::readVariableFromMatFile)
    //     .def_static("read_variable_from_mat_file_py", &MatlabHelper::readVariableFromMatFilePy);

    // expose ObjectiveFunction
    py::class_<ObjectiveFunction, std::shared_ptr<ObjectiveFunction>>(m, "ObjectiveFunction")
        .def(py::init<size_t, const std::string&, const std::string&, const std::string&, bool, CppAdInterface::ModelInfoLevel, ObjectiveFunction::SpecifiedFunctionLevel>(),
            py::arg("variableDim"),
            py::arg("modelName"),
            py::arg("folderName"),
            py::arg("functionName"),
            py::arg("regenerateLibrary") = false,
            py::arg("infoLevel") = CppAdInterface::ModelInfoLevel::SECOND_ORDER,
            py::arg("specifiedFunctionLevel") = ObjectiveFunction::SpecifiedFunctionLevel::NONE
            )
        
        .def(py::init<size_t, size_t, const std::string&, const std::string&, const std::string&, bool, CppAdInterface::ModelInfoLevel, ObjectiveFunction::SpecifiedFunctionLevel>(),
            py::arg("variableDim"),
            py::arg("parameterDim"),
            py::arg("modelName"),
            py::arg("folderName"),
            py::arg("functionName"),
            py::arg("regenerateLibrary") = false,
            py::arg("infoLevel") = CppAdInterface::ModelInfoLevel::SECOND_ORDER,
            py::arg("specifiedFunctionLevel") = ObjectiveFunction::SpecifiedFunctionLevel::NONE
            ); 
        

    // expose ConstraintFunction
    py::class_<ConstraintFunction, std::shared_ptr<ConstraintFunction>>(m, "ConstraintFunction")
        .def(py::init<size_t, const std::string&, const std::string&, const std::string&, bool, CppAdInterface::ModelInfoLevel, ConstraintFunction::SpecifiedFunctionLevel>(),
            py::arg("variableDim"),
            py::arg("modelName"),
            py::arg("folderName"),
            py::arg("functionName"),
            py::arg("regenerateLibrary") = false,
            py::arg("infoLevel") = CppAdInterface::ModelInfoLevel::FIRST_ORDER,
            py::arg("specifiedFunctionLevel") = ConstraintFunction::SpecifiedFunctionLevel::NONE
        )        

        .def(py::init<size_t, size_t, const std::string&, const std::string&, const std::string&, bool, CppAdInterface::ModelInfoLevel, ConstraintFunction::SpecifiedFunctionLevel>(),
            py::arg("variableDim"),
            py::arg("parameterDim"),
            py::arg("modelName"),
            py::arg("folderName"),
            py::arg("functionName"),
            py::arg("regenerateLibrary") = false,
            py::arg("infoLevel") = CppAdInterface::ModelInfoLevel::FIRST_ORDER,
            py::arg("specifiedFunctionLevel") = ConstraintFunction::SpecifiedFunctionLevel::NONE)


        
        .def(py::init<size_t, const casadi::Function&, const casadi::Function&, CDFParameters&, const std::string&, ConstraintFunction::SpecifiedFunctionLevel>(),
            py::arg("variableDim"),
            py::arg("cdfValueFunction"),
            py::arg("cdfGradientFunction"),
            py::arg("cdfParameters"),
            py::arg("functionName"),
            py::arg("specifiedFunctionLevel") = ConstraintFunction::SpecifiedFunctionLevel::GRADIENT)
        .def(py::init<size_t, const std::string&, const std::string&, CDFParameters&, const std::string&, ConstraintFunction::SpecifiedFunctionLevel>(),
            py::arg("variableDim"),
            py::arg("cdfValueFunctionStr"),   // 用字符串
            py::arg("cdfGradientFunctionStr"),
            py::arg("cdfParameters"),
            py::arg("functionName"),
            py::arg("specifiedFunctionLevel") = ConstraintFunction::SpecifiedFunctionLevel::GRADIENT);

        // expose CDFParameters
        py::class_<CDFParameters>(m, "CDFParameters")
        .def(py::init<
            const std::vector<vector_t>&,
            const std::vector<std::vector<int>>&,
            const size_t,
            const size_t,
            const scalar_t
        >(),
            py::arg("obs_lists"),
            py::arg("obs_index_mapping"),
            py::arg("preset_batch"),
            py::arg("num_configuration_variables"),
            py::arg("bias_z") = 0.5
        )
        .def("getObsLists", &CDFParameters::getObsLists)
        .def("getObsIndexMapping", &CDFParameters::getObsIndexMapping)
        .def("getBatchSize", &CDFParameters::getBatchSize)
        .def("getPresetBatchSize", &CDFParameters::getPresetBatchSize)
        .def("getBiasHeight", &CDFParameters::getBiasHeight)
        .def("getNumNonZerosJacobian", &CDFParameters::getNumNonZerosJacobian)
        .def("getOuterIndex", &CDFParameters::getOuterIndex)
        .def("getInnerIndices", &CDFParameters::getInnerIndices)
        .def("getNumConfigurationVariablesPerStep", &CDFParameters::getNumConfigurationVariablesPerStep)
        .def("changeObsAssign", &CDFParameters::changeObsAssign)
        ; 
}