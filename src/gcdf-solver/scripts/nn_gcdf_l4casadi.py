import torch
import os
import l4casadi as l4c
import time
import casadi as cs

if __name__ == "__main__":
    print(cs.__file__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = torch.load('./model_gcdf_simulation.pth')
    net.eval()
    N = 8000 # choose your own batch size for your problem setting.
    l4c_model = l4c.L4CasADi(net,device='cuda',batched=True, name='cdf')
    num_input = 3 + 6 + 3
    # generate function handle for batch size 8000
    x_sym = cs.SX.sym('x', num_input, N)
    y_sym = l4c_model(x_sym.T)
    f = cs.Function('y', [x_sym], [y_sym])
    df = cs.Function('dy', [x_sym], [cs.jacobian(y_sym, x_sym)])
    jac_y_x = cs.jacobian(y_sym, x_sym)
    x_sym_truncated = cs.vertcat(
        x_sym[3, :],
        x_sym[4, :],
        x_sym[5, :],
        x_sym[6, :],
        x_sym[7, :],
        x_sym[8, :],
        x_sym[9, :],
        x_sym[10, :],
        x_sym[11, :]
    )  # (9, N)
    x_sym_truncated_flat = x_sym_truncated.reshape((-1, 1))  # (9*N, 1)
    x_sym_flat = x_sym.reshape((-1, 1))  # (num_input * N, 1)
    jac_trunc_x = cs.jacobian(x_sym_truncated_flat, x_sym_flat)  # (9 * N, num_input * N)
    jac_y_trunc = jac_y_x @ jac_trunc_x.T  # (N, num_input * N) @ (num_input * N, 9 * N) = (N, 9 * N)
    df = cs.Function('df', [x_sym], [jac_y_x]) 
    df_truncated = cs.Function('df_test_truncated', [x_sym], [jac_y_trunc]) # only calculate a partial derivative with respect to q
    # # save to .so
    opts = {"cpp": False, "with_header": True}
    prefix_code = os.path.abspath("../core/")
    os.makedirs(prefix_code, exist_ok=True)
    # x = cs.SX.sym('x')
    # ff = cs.Function('f', [x], [x**2])
    # cg = cs.CodeGenerator("simple.c", opts)
    # cg.add(ff)
    cg = cs.CodeGenerator("gcdf_func.c", opts)
    cg.add(f)
    cg.add(df_truncated)
    cg.generate(os.path.join(prefix_code, ""))

#     # prefix_lib = os.path.abspath("./function_lib/")
#     # os.makedirs(prefix_lib, exist_ok=True)




#     print(f'L4CASADI_LIB_DIR: {os.path.dirname(os.path.abspath(l4c.__file__))}/lib')
#     print(f'TORCH_LIB_DIR: {os.path.dirname(os.path.abspath(torch.__file__))}/lib')
#     print(f'L4CASADI_GEN_LIB_DIR: {l4c_model.shared_lib_dir}')
#     # exit()
#     compile_command = (
#     f"gcc -fPIC -shared -O3 {prefix_code}/cdf_func.c -o {prefix_lib}/lib_cdf_func.so "
#     f"-I{L4CASADI_GEN_LIB_DIR} "
#     f"-L{L4CASADI_LIB_DIR} -L{L4CASADI_GEN_LIB_DIR} -L{TORCH_LIB_DIR} "
#     f"-ll4casadi -lcdf"
# )
#     print(compile_command)
#     compile_flag = os.system(compile_command)
#     assert compile_flag == 0, "Compilation failed!"
#     print("Compilation succeeded!")

    # use casadi external to load the shared library
    # f_ext = cs.external('y', os.path.join(prefix_lib, 'lib_cdf_func.so'))
    # print("f_ext:", f_ext(x_batch_vcat.T))
