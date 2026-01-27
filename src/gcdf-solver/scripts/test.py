import casadi as cs
import sys
sys.path.append("/home/robot/workspace/src/gcdf-solver/lib")
import pyCRISP

# CasADi configuration
casadi_lib_path = "/home/robot/workspace/src/gcdf-solver/lib/libcdf_func.so"
cdf_value_function = cs.external("y", casadi_lib_path)
cdf_gradient_function = cs.external("df_test_truncated", casadi_lib_path)

# CasADi function warmup
warmup_x = cs.DM.zeros(*cdf_value_function.sparsity_in(0).shape)
for _ in range(1):
    print(cdf_value_function(warmup_x))
    print(cdf_gradient_function(warmup_x))
