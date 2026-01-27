

import torch
import os
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
import numpy as np
import sys
sys.path.append(os.path.join(CUR_DIR,'./resource/'))
from robot_layer.moma_layer_pk import MoMaLayer
from bf_sdf import BPSDF
from torchmin import minimize
import time
import math
import copy

PI = math.pi

class DataGenerator():
    def __init__(self,device):
        # robot model
        self.robot = MoMaLayer(device)
        self.bp_sdf = BPSDF(8,-1.0,1.0,self.robot,device)
        print("robot model loaded")
        self.model = torch.load(os.path.join(CUR_DIR,'./resource/models/BP_8.pt'),weights_only=False)
        print("sdf model loaded")
        print(self.model.keys())
        self.q_max = self.robot.q_max
        self.q_min = self.robot.q_min
        print(self.q_max)
        print(self.q_min)

        # device
        self.device = device

        # data generation
        self.workspace = [[-1.0,-1.0,0.0],[1.0,1.0,1.5]]
        self.n_discrete = 40       # total number of x: n_discrete**3
        self.batchsize = 10000      # batch size of q
        # self.pose = torch.eye(4).unsqueeze(0).to(self.device).expand(self.batchsize,4,4).float()
        self.epsilon = 1e-3         # distance threshold to filter data
        self.threshold = 0.0
        self.threshold = torch.tensor(self.threshold).to(self.device)

    def compute_sdf(self,x,q,return_index = False):
        # x : (Nx,3)
        # q : (Nq,9)
        # return_index : if True, return the index of link that is closest to x
        # return d : (Nq)
        # return idx : (Nq) optional
        q_t = q[:,:2]
        q_r = q[:,2:]
        q_t_0 = torch.zeros_like(q_t).to(self.device)
        pose = torch.eye(4).unsqueeze(0).to(self.device).expand(len(q),4,4).float()
        if not return_index:
            # d,_ = self.bp_sdf.get_whole_body_sdf_batch_xy(x,pose, q_r,q_t, self.model,use_derivative =False)
            d,_ = self.bp_sdf.get_whole_body_sdf_batch_xy(x,pose, q_r,q_t_0, self.model,use_derivative =False)
            d = d.min(dim=1)[0]
            return d
        else:
            # d,_,idx = self.bp_sdf.get_whole_body_sdf_batch_xy(x,pose, q_r,q_t, self.model,use_derivative =False,return_index = True)
            d,_,idx = self.bp_sdf.get_whole_body_sdf_batch_xy(x,pose, q_r,q_t_0, self.model,use_derivative =False,return_index = True)
            d,pts_idx = d.min(dim=1)
            idx = idx[torch.arange(len(idx)),pts_idx]
            print("d",d, "idx",idx)
            return d,idx

    def given_x_find_q(self,x,q = None, batchsize = None,return_mask = False,epsilon = 1e-3):
        # x : (N,3)
        # scale x to workspace
        if not batchsize:
            batchsize = self.batchsize

        def cost_function(q):
            #  find q that d(x,q) = 0
            # q : B,2
            # x : N,3

            d = self.compute_sdf(x,q)
            cost = torch.sum((d - self.threshold)**2)
            return cost
        
        t0 = time.time()
        # optimizer for data generation
        if q is None:
            q = torch.rand(batchsize,9).to(self.device)*(self.q_max-self.q_min)+self.q_min
            q[:,:2] = q[:,:2] * 0.0
        res = minimize(
            cost_function, 
            q, 
            method='l-bfgs', 
            options=dict(line_search='strong-wolfe'),
            max_iter=50,
            disp=0
            )
        
        d,idx = self.compute_sdf(x,res.x,return_index=True)
        d,idx = d.squeeze(),idx.squeeze()

        mask = torch.abs(d - self.threshold) < epsilon
        # q_valid,d,idx = res.x[mask],d[mask],idx[mask]
        boundary_mask = ((res.x > self.q_min) & (res.x < self.q_max)).all(dim=1)
        final_mask = mask & boundary_mask
        final_q,idx = res.x[final_mask],idx[final_mask]
        # q0 = q0[mask][boundary_mask]
        print("point: ", x)
        print('number of q_valid: \t{} \t time cost:{}'.format(len(final_q),time.time()-t0))
        if return_mask:
            return final_mask,final_q,idx
        else:
            return final_q,idx

    def distance_q(self,x,q):
        # x : (Nx,3)
        # q : (Np,7)
        # return d : (Np) distance between q and x in C space. d = min_{q*}{L2(q-q*)}. sdf(x,q*)=0

        # compute d
        Np = q.shape[0]
        q_template,link_idx = self.given_x_find_q(x)
        print(q_template.shape)

        if link_idx.min() == 0:
            return torch.zeros(Np).to(self.device)
        else:
            link_idx[link_idx==7] = 6
            link_idx[link_idx==8] = 7
            d = torch.inf*torch.ones(Np,7).to(self.device)
            for i in range(link_idx.min(),link_idx.max()+1):
                mask = (link_idx==i)
                d_norm = torch.norm(q[:,:i].unsqueeze(1)- q_template[mask][:,:i].unsqueeze(0),dim=-1)
                d[:,i-1] = torch.min(d_norm,dim=-1)[0]
        d = torch.min(d,dim=-1)[0]

        # compute sign of d
        d_ts = self.compute_sdf(x,q)
        mask =  (d_ts < 0)
        d[mask] = -d[mask]
        return d 

    def projection(self,x,q):
        q.requires_grad = True
        d = self.distance_q(x,q)
        grad = torch.autograd.grad(d,q,torch.ones_like(d),create_graph=True)[0]
        q_new = q - grad*d.unsqueeze(-1)
        return q_new

    def generate_offline_data(self,save_path = CUR_DIR):
        
        x = torch.linspace(self.workspace[0][0],self.workspace[1][0],self.n_discrete).to(self.device)
        y = torch.linspace(self.workspace[0][1],self.workspace[1][1],self.n_discrete).to(self.device)
        z = torch.linspace(self.workspace[0][2],self.workspace[1][2],self.n_discrete).to(self.device)
        x,y,z = torch.meshgrid(x,y,z)
        pts = torch.stack([x,y,z],dim=-1).reshape(-1,3)
        data = {}
        for i,p in enumerate(pts):
            q,idx = self.given_x_find_q(p.unsqueeze(0)) 
            data[i] ={
                'x':    p.detach().cpu().numpy(),
                'q':    q.detach().cpu().numpy(),
                'idx':  idx.detach().cpu().numpy(),
            }
            print(f'point {i} finished, number of q: {len(q)}')
        np.save(os.path.join(save_path,'data.npy'),data)

def analysis_data(x):
    # Compute the squared Euclidean distance between each row
    diff = x.unsqueeze(1) - x.unsqueeze(0)
    diff = diff.pow(2).sum(-1)

    # Set the diagonal elements to a large value to exclude self-distance
    diag_indices = torch.arange(x.shape[0])
    diff[diag_indices, diag_indices] = float('inf')
    
    # Compute the Euclidean distance by taking the square root
    diff = diff.sqrt()
    min_dist = torch.min(diff,dim=1)[0]
    print(f'distance\tmax:{min_dist.max()}\tmin:{min_dist.min()}\taverage:{min_dist.mean()}')



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    gen = DataGenerator(device)
    # x = torch.tensor([[0.5,0.5,0.5]]).to(device)
    # gen.single_point_generation(x)
    gen.generate_offline_data()

    x_range = torch.arange(0.0, 0.5, 0.1)  # From -1 to 1, step size 0.25
    y_range = torch.tensor([0.0])   

        # Generate mesh grid
    grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='ij')

        # Flatten to point list
    points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

        # The third column is all 1.0
    heights = torch.ones(points.size(0), 1) * 1.0
    x  = torch.cat([points, heights], dim=1).to(device)
    x.requires_grad_()  # Keep gradient tracking
    q = []
    for pt in x:
        q_i , idx = gen.given_x_find_q(pt, batchsize=20000, return_mask=False, epsilon=1e-2)
        q.append(q_i)
        d_c = torch.zeros(q_i.shape[0]).to(device)
        for i in range(q_i.shape[0]):
            d_c[i] = torch.norm(q_i[i,:idx[i]+2])
            # d_c[i] = torch.norm(q_i[i,:])
        closest_q = q_i[d_c.argmin()]
        d_min = d_c.min()
        print(f"point {pt} has {len(q_i)} q, closest distance to origin is {d_min}, corresponding q is {closest_q}")
        # print(q_norms[:100])
        import trimesh
        scene = trimesh.Scene()
        pose = torch.eye(4).unsqueeze(0).to(device).float()
        scene.add_geometry(trimesh.PointCloud(pt.detach().cpu().numpy().reshape(1, 3), colors=[255, 0, 0]))

        robot_mesh0 = gen.robot.get_forward_robot_mesh_xy(pose, closest_q.unsqueeze(0))
        robot_mesh0 = np.sum(robot_mesh0)
        robot_mesh0.visual.face_colors = [0, 255, 0, 100]
        scene.add_geometry(robot_mesh0)
        
        q0 = torch.zeros(9).to(device)
        robot_mesh1 = gen.robot.get_forward_robot_mesh_xy(pose, q0.unsqueeze(0))
        robot_mesh1 = np.sum(robot_mesh1)
        robot_mesh1.visual.face_colors = [0, 0, 255, 100]
        scene.add_geometry(robot_mesh1)

        scene.show()

    print(q.shape)