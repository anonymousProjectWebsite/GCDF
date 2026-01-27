
# robot layer implementation using pytorch kinematics
import torch
import trimesh
import glob
import os
import numpy as np
import pytorch_kinematics as pk
import copy
CUR_PATH = os.path.dirname(os.path.realpath(__file__))

def save_to_mesh(vertices, faces, output_mesh_path=None):
    assert output_mesh_path is not None
    with open(output_mesh_path, 'w') as fp:
        for vert in vertices:
            fp.write('v %f %f %f\n' % (vert[0], vert[1], vert[2]))
        for face in faces+1:
            fp.write('f %d %d %d\n' % (face[0], face[1], face[2]))
    print('Output mesh save to: ', os.path.abspath(output_mesh_path))

class  MoMaLayer(torch.nn.Module):
    def __init__(self, device='cpu', mesh_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"./meshes/moma/*.STL")):
        super().__init__()
        dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.device = device
        self.urdf_path = os.path.join(dir_path, 'urdf/moma.urdf')
        self.mesh_path = mesh_path
        self.chain = pk.build_serial_chain_from_urdf(open(self.urdf_path).read().encode(),"armend_effector_link").to(dtype = torch.float32,device = self.device)
        self.links = self.chain.get_link_names()[1:]
        print("self.links: ", self.links)
        joint_lim = torch.tensor(self.chain.get_joint_limits())
        print("joint limits: ", joint_lim)
        self.theta_min = joint_lim[0,:].to(self.device)
        self.theta_max = joint_lim[1,:].to(self.device)
        self.q_max = torch.cat([torch.tensor([1.0, 1.0 ], device=self.theta_max.device), self.theta_max])
        self.q_min = torch.cat([torch.tensor([-1.0, -1.0], device=self.theta_min.device), self.theta_min])

        self.theta_mid = (self.theta_min + self.theta_max) / 2.0
        self.theta_min_soft = (self.theta_min-self.theta_mid)*0.8 + self.theta_mid
        self.theta_max_soft = (self.theta_max-self.theta_mid)*0.8 + self.theta_mid
        self.dof = len(self.theta_min)
        self.meshes = self.load_meshes()
        for k in self.meshes.keys():
            print("Loaded mesh:", k)

    def load_meshes(self):
        mesh_files = glob.glob(self.mesh_path)
        
        mesh_files = [f for f in mesh_files if os.path.isfile(f)]
        meshes = {}

        for mesh_file in mesh_files:
            # if self.mesh_path.split('/')[-2]=='visual':
            #     name = os.path.basename(mesh_file)[:-4].split('_')[0]
            # else:
            #     name = os.path.basename(mesh_file)
            name = os.path.basename(mesh_file)[:-4]

            mesh = trimesh.load(mesh_file, force='mesh')
            meshes[name] = mesh
        return meshes

    def forward_kinematics(self, theta):
        ret = self.chain.forward_kinematics(theta, end_only=False)
        transformations = {}
        # print(ret.keys())
        for k in ret.keys():
            trans_mat = ret[k].get_matrix()
            transformations[k] = trans_mat
        return transformations
    
    def theta2mesh(self, theta):
        trans = self.forward_kinematics(theta)
        robot_mesh = []
        for k in self.meshes.keys():
            if k !='finger':
                mesh = copy.deepcopy(self.meshes[k])
                vertices = torch.from_numpy(mesh.vertices).to(self.device).float()
                vertices = torch.cat([vertices, torch.ones([vertices.shape[0], 1], device=self.device)], dim=-1).t()
                transformed_vertices = torch.matmul(trans[k].squeeze(), vertices).t()[:, :3].detach().cpu().numpy()
                mesh.vertices = transformed_vertices
                robot_mesh.append(mesh)
        return robot_mesh
    
    def get_transformations_each_link(self,pose, theta):
        # pose: (1,4,4)
        # theta: (N,7)
        transformations = self.forward_kinematics(theta)
        link_transforms = []
        links = self.links
        for k in transformations.keys():
            if k in links:
                link_transforms.append(transformations[k])
        
        return link_transforms

    def get_forward_robot_mesh(self, pose, theta):
        return self.theta2mesh(theta)
    
    def get_forward_robot_mesh_xy(self, pose, q):
        # pose: (1,4,4)
        # q_r: (N,7)
        q_t = q[:,:2]
        q_r = q[:,2:]
        trans = self.forward_kinematics(q_r)
        for k in trans.keys():
            trans[k][:,0,3] += q_t[:,0]
            trans[k][:,1,3] += q_t[:,1]

        robot_mesh = []
        for k in self.meshes.keys():
            if k !='finger':
                mesh = copy.deepcopy(self.meshes[k])
                vertices = torch.from_numpy(mesh.vertices).to(self.device).float()
                vertices = torch.cat([vertices, torch.ones([vertices.shape[0], 1], device=self.device)], dim=-1).t()

                transformed_vertices = torch.matmul(trans[k].squeeze(), vertices).t()[:, :3].detach().cpu().numpy()
                mesh.vertices = transformed_vertices
                robot_mesh.append(mesh)
        return robot_mesh



if __name__ == "__main__":
    device = 'cuda'
    moma = MoMaLayer(device).to(device)
    scene = trimesh.Scene()
    theta = torch.tensor([0.0, -1.7370, -0.0455,  0.1577, -2.8271, -2.0578,  1.8342]).float().to(device).reshape(-1,7)
    pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).expand(len(theta),-1,-1).float()

    trans = moma.forward_kinematics(theta)
    transformations = moma.get_transformations_each_link(pose, theta)
    print("transformations:  shape", transformations[0].shape)
    # print(trans)
    # pts = np.array([trans[k][0,:3,3].detach().cpu().numpy() for k in trans.keys()])
    # print(pts)
    # PC = trimesh.points.PointCloud(pts, colors=[255,0,0])
    # scene.add_geometry(PC)
    robot_mesh = moma.theta2mesh(theta)
    scene.add_geometry(robot_mesh)
    scene.show()