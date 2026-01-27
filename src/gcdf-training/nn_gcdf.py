
import numpy as np
import os
import sys
import torch
import math
import time
import threading

CUR_PATH = os.path.dirname(os.path.realpath(__file__))
from mlp import MLPRegression
sys.path.append(os.path.join(CUR_PATH,'./resource/'))
from robot_layer.moma_layer_pk import MoMaLayer
import bf_sdf
import wandb

PI = math.pi
np.random.seed(10)

class CDF:
    def __init__(self,device) -> None:
        # device
        self.device = device  
        self.gpulist =[0]

        self.batch_x = 20
        self.batch_q = 200
        self.max_q_per_link = 200
        self.batch_x_split = 5
        # uncomment these lines to process the generated data and train your own CDF
        self.raw_data = np.load(os.path.join(CUR_PATH,'data.npy'),allow_pickle=True).item()
        self.process_data(self.raw_data)

        self.data_path = os.path.join(CUR_PATH,'data.pt') 
        print(f"Loading data from {self.data_path}")
        self.data = self.load_data(self.data_path)
        self.len_data = len(self.data['k'])
        self.weights= torch.tensor([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(self.device)
        print(f"Data loaded, length: {self.len_data}")
         

        # panda robot
        self.panda = MoMaLayer(device)

        self.bp_sdf = bf_sdf.BPSDF(8,-1.0,1.0,self.panda,device)
        self.bp_sdf_model = torch.load(os.path.join(CUR_PATH,'./resource/models/BP_8.pt'),map_location=self.device)

        self.w0 = 5.0
        self.w1 = 0.1
        self.w2 = 0.01
        self.w3 = 0.1
        self.mlp_layers = [1024,1024,512, 512, 256, 256, 128, 128]

         # init wandb
         # Initialize wandb

    def _farthest_point_sampling(self, points, k):
        # points: (N, D)
        # returns: (k, D)
        n = points.shape[0]
        if k >= n:
            return points
        indices = torch.zeros(k, dtype=torch.long, device=points.device)
        indices[0] = torch.randint(0, n, (1,), device=points.device)
        dist = torch.cdist(points[indices[0]].unsqueeze(0), points).squeeze(0)
        for i in range(1, k):
            indices[i] = torch.argmax(dist)
            new_dist = torch.cdist(points[indices[i]].unsqueeze(0), points).squeeze(0)
            dist = torch.minimum(dist, new_dist)
        return points[indices]
        

    def process_data(self,data):
        keys = list(data.keys())  # Create a copy of the keys
        processed_data = {}
        count = 0
        for k in keys:
            if len(data[k]['q']) == 0:
                data.pop(k)
                continue
            else:
                print(f"Processing key: {k} with {len(data[k]['q'])} samples")
            q = torch.from_numpy(data[k]['q']).float().to(self.device)
            q_idx = torch.from_numpy(data[k]['idx']).float().to(self.device)
            # q_idx[q_idx==0] = 8
            q_idx[q_idx==1] = 0
            q_idx[q_idx==2] = 1
            q_idx[q_idx==3] = 2
            q_idx[q_idx==4] = 3
            q_idx[q_idx==5] = 4
            q_idx[q_idx==6] = 5
            q_idx[q_idx==7] = 6
            # q_idx[q_idx==8] = 7

            q_lib = torch.inf*torch.ones(self.max_q_per_link,9,7).to(self.device)
            for i in range(0,7):
                mask = (q_idx==i)
                if len(q[mask])>self.max_q_per_link:
                    fps_q = self._farthest_point_sampling(q[mask], self.max_q_per_link)
                    q_lib[:,:,i] = fps_q
                    # print(q_lib[:,:,i]) 
                elif len(q[mask])>0:
                    q_lib[:len(q[mask]),:,i] = q[mask]
                
                print("len q[mask]",len(q[mask]),"max_q_per_link",self.max_q_per_link)

            processed_data[k] = {
                'x':torch.from_numpy(data[k]['x']).float().to(self.device),
                'q':q_lib,
            }
            count += 1


        final_data = {
            'x': torch.cat([processed_data[k]['x'].unsqueeze(0) for k in processed_data.keys()],dim=0),
            'q': torch.cat([processed_data[k]['q'].unsqueeze(0) for k in processed_data.keys()],dim=0),
            'k':torch.tensor([k for k in processed_data.keys()]).to(self.device)
        }

        # del processed_data
        # torch.cuda.empty_cache()

        final_data['unique_z'] = torch.unique(final_data['x'][:, 2])
        z_indices = []
        for z in final_data['unique_z']:
            indices = (final_data['x'][:, 2] == z).nonzero(as_tuple=True)[0]
            z_indices.append(indices)

        M_max = max(len(idx) for idx in z_indices)
        idx_pad = torch.full((len(z_indices), M_max), -1, dtype=torch.long, device=self.device)
        for i, idx in enumerate(z_indices):
            idx_pad[i, :len(idx)] = torch.tensor(idx, device=self.device)
        final_data['z_indices'] = idx_pad
    
        torch.save(final_data,os.path.join(CUR_PATH,'data.pt'))

        return data
    
    def load_data(self,path):
        data = torch.load(path,map_location=self.device)
        return data

    def select_data(self):
        # x_batch:(batch_x,3)
        # q_batch:(batch_q,7)
        # d:(batch_x,batch_q)
        
        x = self.data['x']
        q = self.data['q']

        idx = torch.randint(0,len(x),(self.batch_x,)) 
        # idx = torch.tensor([4000])
        # x_batch,q_lib = x[idx],q[idx]
        x_batch = x[idx]
        q_lib = self.get_qlib_by_x_batch_extreme(x_batch,idx)
        # q_lib = q[idx]
        # print(x_batch)
        q_batch = self.sample_q()   
        d,grad = self.parallel_decode_distance(q_batch,q_lib,x_batch)
        return x_batch,q_batch,d,grad

    
    def get_qlib_by_x_batch_extreme(self, x_in_batch,index):
        """
        Inputs:
            final_data: dict, contains
                'unique_z': (Z,)
                'z_indices': (Z, M_max) int, padded with -1
                'q': (N_all, max_q, 9, 7)
                'x': (N_all, 3)
            x_in_batch: (N, 3)
        Outputs:
            qlib_batch: (N, M_max * max_q, 9, 7)  # padded with inf
        """

        device = x_in_batch.device
        N = x_in_batch.shape[0]
        x = self.data['x']
        q = self.data['q']
        max_q = self.max_q_per_link
        masks = []
        x_range = 0.5
        y_range = 0.5
        for pt in x_in_batch:
            mask = (
                    torch.isclose(x[:, 2], torch.tensor(pt[2])) &
                    (x[:, 0] >= pt[0] -x_range) & (x[:, 0] <= pt[0] + x_range) &
                    (x[:, 1] >= pt[1]- y_range) & (x[:, 1] <= pt[1] + y_range)
                )
            masks.append(mask)
            

        indices_list = [mask.nonzero(as_tuple=True)[0] for mask in masks]
        M_max = max(len(idx) for idx in indices_list)
        indices = torch.full((len(indices_list), M_max), -1, dtype=torch.long, device=device)
        for i, idx in enumerate(indices_list):
            indices[i, :len(idx)] = idx.to(device)

        gather_idx = indices  # (N, M_max)
        valid_mask = gather_idx != -1  # (N, M_max)
        # 3. gather x_targets
        x_targets = torch.full((N, M_max, 3), float('inf'), device=device)
        x_targets[valid_mask] = x[gather_idx[valid_mask]]

        # 4. xy_diff
        xy_diff =  x_in_batch[:, :2].unsqueeze(1) - x_targets[:, :, :2]  # (N, M_max, 2)
        # 5. gather q
        q_pad = torch.full((N, M_max, max_q, 9, 7), float('inf'), device=device)
        # Use double indexing here to skip -1 entries
        for i in range(N):
            valid = valid_mask[i]
            idxs = gather_idx[i][valid]
            if idxs.numel() > 0:
                q_pad[i, valid] = q[idxs]
                # print("x_in_batch", x_in_batch[i])
        # 6. Update q_pad[..., 0], q_pad[..., 1]
        q_pad[:, :, :, :2, :] = xy_diff.unsqueeze(2).unsqueeze(-1)

        # 7. reshape and merge
        qlib_batch = q_pad.reshape(N, M_max * max_q, 9, 7)
        return qlib_batch
    # def decode_distance(self,q_batch,q_lib,x_batch):
    #     # batch_q:(batch_q,7) /100 ，7
    #     # q_lib:(batch_x,self.max_q_per_link,9,7)
    #     # print("q_lib shape:", q_lib.shape)
    #     # print("q_batch shape:", q_batch.shape)
    #     batch_x = q_lib.shape[0]
    #     batch_q = q_batch.shape[0]
    #     d_tensor = torch.ones(batch_x,batch_q,9).to(self.device)*torch.inf
    #     # d_tensor = torch.ones(batch_x, batch_q, 7).to(self.device) * 1000
       
    #     grad_tensor  = torch.zeros(batch_x,batch_q,9,7).to(self.device)
    #     for i in range(2,9):
    #         q_lib_temp = q_lib[:,:,:i+1,i-2].reshape(batch_x*self.max_q_per_link,-1).unsqueeze(0).expand(batch_q,-1,-1)
    #         q_batch_temp = q_batch[:,:i+1].unsqueeze(1).expand(-1,batch_x*self.max_q_per_link,-1)
    #         q_lib_temp_weighted = q_lib_temp * self.weights[:i+1].unsqueeze(0).unsqueeze(0)
    #         q_batch_temp_weighted = q_batch_temp * self.weights[:i+1].unsqueeze(0).unsqueeze(0)
    #         d_norm = torch.norm((q_lib_temp_weighted - q_batch_temp_weighted),dim=-1).reshape(batch_q,batch_x,self.max_q_per_link)
    #         d_norm_min,d_norm_min_idx = d_norm.min(dim=-1)
    #         grad = torch.autograd.grad(d_norm_min.reshape(-1),q_batch_temp,torch.ones_like(d_norm_min.reshape(-1)),retain_graph=True)[0]
    #         grad_min_q = grad.reshape(batch_q,batch_x,self.max_q_per_link,-1).gather(2,d_norm_min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,i+1))[:,:,0,:]
    #         grad_tensor[:,:,:i+1,i-2] = grad_min_q.transpose(0,1)
    #         d_tensor[:,:,i-2] = d_norm_min.transpose(0,1)
    #     d,d_min_idx = d_tensor.min(dim=-1)
    #     assert not torch.isnan(d).any(), "d contains NaN"
    #     assert not torch.isinf(d).any(), "d contains Inf"
    #     grad_final = grad_tensor.gather(3,d_min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,9,7))[:,:,:,0]
    #     print("d shape:", d.shape, "grad_final shape:", grad_final.shape)
    #     return d, grad_final

    def decode_distance(self,q_batch,q_lib,x_batch):
        # batch_q:(batch_q,7) /100 ，7
        # q_lib:(batch_x,self.max_q_per_link,9,7)
        # print("q_lib shape:", q_lib.shape)
        # print("q_batch shape:", q_batch.shape)
        device = q_lib.device

        batch_x = q_lib.shape[0]
        batch_q = q_batch.shape[0]
        q_per_link = q_lib.shape[1]
        
        d_tensor = torch.ones(batch_x,batch_q,7).to(device)*torch.inf
        d_norm_min_idx_tensor = torch.zeros(batch_x,batch_q,7).to(device).long()
        grad_tensor = torch.zeros(batch_x,batch_q,9,7).to(device)
        weights = self.weights.to(device)
        # split x
        batch_x_split = self.batch_x_split
        
        
        for i in range(2, 9):
            weights_broadcast = weights[:i+1].view(1, 1, -1)
            
            for x_start in range(0, batch_x, batch_x_split):
                x_end = min(x_start + batch_x_split, batch_x)
                q_lib_slice = q_lib[x_start:x_end]  # take a small slice
                bx = x_end - x_start

                with torch.no_grad():
                    q_lib_temp = q_lib_slice[:, :, :i+1, i-2].reshape(bx*q_per_link, -1).unsqueeze(0).expand(batch_q, -1, -1)
                    q_batch_temp = q_batch[:, :i+1].unsqueeze(1).expand(-1, bx*q_per_link, -1)
                    diff = (q_lib_temp - q_batch_temp) * weights_broadcast
                    d_norm = torch.norm(diff , dim=-1).reshape(batch_q, bx, q_per_link)
                    d_norm_min, d_norm_min_idx = d_norm.min(dim=-1)  # (batch_q, bx)
                    d_tensor[x_start:x_end, :, i-2] = d_norm_min.transpose(0, 1)
                    d_norm_min_idx_tensor[x_start:x_end, :, i-2] = d_norm_min_idx.transpose(0, 1)

                # gradient part
            q_lib_grad = q_lib[:, :, :i+1, i-2].unsqueeze(1).expand(-1, batch_q, -1, -1).gather(
                2, d_norm_min_idx_tensor[:, :, i-2].unsqueeze(-1).expand(-1, -1, i+1).unsqueeze(2)
            )[:, :, 0, :]
            q_batch_grad = q_batch[:, :i+1].unsqueeze(0).expand(batch_x, -1, -1)
            q_batch_grad.requires_grad_(True)  # requires gradients
            d_norm_min_val = (q_lib_grad * weights[:i+1].unsqueeze(0).unsqueeze(0) -
                            q_batch_grad * weights[:i+1].unsqueeze(0).unsqueeze(0)).norm(dim=-1)
            grad = torch.autograd.grad(d_norm_min_val.reshape(-1), q_batch_grad, torch.ones_like(d_norm_min_val.reshape(-1)), retain_graph=False)[0]
            grad_tensor [:, :, :i+1, i-2] = grad.reshape(batch_x, batch_q, i+1)

            # del q_lib_temp, q_batch_temp, d_norm, d_norm_min, d_norm_min_idx, q_lib_grad, q_batch_grad, grad
            # torch.cuda.empty_cache()

            # compute gradient use d_norm_min_idx
        d,d_min_idx = d_tensor.min(dim=-1)
        grad_final = grad_tensor.gather(3,d_min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,9,7))[:,:,:,0]

        # assert not torch.isnan(d).any(), "d contains NaN"
        # assert not torch.isinf(d).any(), "d contains Inf"
        

        return d, grad_final

    def worker_decode_distance(self, rank, x_batch, q_batch, q_lib, device):
        # rank: GPU index
        # device: 'cuda:0', 'cuda:1', ...
        x_batch = x_batch.to(device)
        x_batch.requires_grad_()
        q_batch = q_batch.to(device)
        q_batch.requires_grad_()
        q_lib = q_lib.to(device)
        q_batch.requires_grad_()
        # Assume CDF is globally available
        d, grad = self.decode_distance(q_batch, q_lib, x_batch)
        return d, grad

    def parallel_decode_distance(self, q_batch, q_lib, x_batch):
        num_gpus = len(self.gpulist)
        # print(f"Using {num_gpus} GPUs for parallel decoding.")
        splits_x = torch.chunk(x_batch, num_gpus, dim=0)
        splits_q_lib = torch.chunk(q_lib, num_gpus, dim=0)
        d_list, grad_list = [None]*num_gpus, [None]*num_gpus

        def worker(rank, x_sub, q_batch, q_lib_sub, device):
            d, grad = self.worker_decode_distance(rank, x_sub, q_batch, q_lib_sub, device)
            d_list[rank] = d.cpu()
            grad_list[rank] = grad.cpu()


        threads = []
        for rank in range(num_gpus):
            x_sub = splits_x[rank]
            q_lib_sub = splits_q_lib[rank]
            device = f'cuda:{self.gpulist[rank]}'
            t = threading.Thread(target=worker, args=(rank, x_sub, q_batch, q_lib_sub, device))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        
        d_all = torch.cat(d_list, dim=0)
        grad_all = torch.cat(grad_list, dim=0)

        
        sign = self.get_sign(x_batch, q_batch)
        d_final = d_all.to(self.device) * sign
        grad_final = grad_all.to(self.device) * sign.unsqueeze(-1)

        return d_final, grad_final

        # Follow-up stays the same as your original code
    def get_sign(self, x, q):
        # x : (Nx,3)
        # q : (Np,7)
        d_ts = self.compute_sdf(x, q)
        mask = (d_ts <= 0.05)
        sign = torch.ones_like(d_ts).to(self.device)
        sign[mask] = -1
        mask = (d_ts <= -0.05) 
        sign[mask] = -3


        # (nx, nq) -> (nq, nx)
        sign = sign.transpose(0, 1)
        return sign
    
    def compute_sdf(self,x,q):
        # get_whole_body_sdf_batch_xy
        device = x.device
        bp_sdf_model = self.bp_sdf_model
        pose = torch.eye(4).unsqueeze(0).expand(len(q),-1,-1).to(device).float()
        sdf,_ = self.bp_sdf.get_whole_body_sdf_batch_xy(x, pose, q[:,2:], q[:,:2], bp_sdf_model,use_derivative=False)

        return sdf

    def sample_q(self,batch_q = None):
        if batch_q is None:
            batch_q = self.batch_q
        q_sampled = self.panda.q_min + torch.rand(batch_q,9).to(self.device)*(self.panda.q_max-self.panda.q_min)
        # q_sampled[:,:2] = q_sampled[:,:2]*0.0
        q_sampled.requires_grad = True
        return q_sampled
    
    def projection(self,q,d,grad):
        q_new = q - grad*d.unsqueeze(-1)*1/(self.weights**2).unsqueeze(0)
        return q_new

    def train_nn(self,epoches=500):
        wandb.login()
        wandb.init(
            project="gcdf-training",  # project name
            name="MLPRegression_training",  # run name
            config={
                "mlp_layers": self.mlp_layers,
                "batch_x": self.batch_x,
                "batch_q": self.batch_q,
                "max_q_per_link": self.max_q_per_link,
                "weights": self.weights.tolist(),
                "w0": self.w0,
                "w1": self.w1,
                "w2": self.w2,
                "w3": self.w3,
                "weight_eikonal": False,
            }
        )

        # model
        # input: [x,q] (B,3+7)

        model = MLPRegression(input_dims=12, output_dims=1, mlp_layers=self.mlp_layers,skips=[], act_fn=torch.nn.GELU, nerf=True)
        # model = torch.nn.DataParallel(model)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000,
                                                        threshold=0.01, threshold_mode='rel',
                                                        cooldown=0, min_lr=0, eps=1e-04, verbose=True)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        # torch.autograd.set_detect_anomaly(True)

        COSLOSS = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        model_dict = {}
        for iter in range(epoches):
            model.train()
            with torch.cuda.amp.autocast():
                start_time = time.time()
                x_batch,q_batch,d,gt_grad = self.select_data()
                select_time = time.time() - start_time

                x_inputs = x_batch.unsqueeze(1).expand(-1,self.batch_q,-1).reshape(-1,3)
                q_inputs = q_batch.unsqueeze(0).expand(self.batch_x,-1,-1).reshape(-1,9)
                
                inputs = torch.cat([x_inputs,q_inputs],dim=-1)
                outputs = d.reshape(-1,1)
                gt_grad = gt_grad.reshape(-1,9)
                weights = torch.ones_like(outputs).to(device)
                # weights = (1/outputs).clamp(0,1)
                
                d_pred = model.forward(inputs)
                d_grad_pred = torch.autograd.grad(d_pred, q_inputs, torch.ones_like(d_pred), retain_graph=True,create_graph=True)[0]
                
                # Compute the Eikonal loss
                # eikonal_loss = torch.abs(d_grad_pred.norm(2, dim=-1) - 1).mean()
                weights =(1/self.weights**2).unsqueeze(0)
                eikonal_loss = torch.abs(torch.sqrt(torch.sum(weights * (d_grad_pred ** 2), dim=-1)) - 1).mean()

                # Compute the tension loss
                dd_grad_pred = torch.autograd.grad(d_grad_pred, q_inputs, torch.ones_like(d_grad_pred), retain_graph=True,create_graph=True)[0]
                # assert not torch.isnan(dd_grad_pred).any(), "dd_grad_pred contains NaN"
                # gradient loss
                gradient_loss = (1 - COSLOSS(d_grad_pred, gt_grad)).mean()
                # tension loss
                tension_loss = dd_grad_pred.square().sum(dim=-1).mean()
                # Compute the MSE loss
                d_loss = ((d_pred-outputs)**2).mean()

                # Combine the two losses with appropriate weights
                w0 = self.w0
                w1 = self.w1
                w2 = self.w2
                w3 = self.w3
                loss = w0 * d_loss + w1 * eikonal_loss + w2 * tension_loss + w3 * gradient_loss

                # # Print the losses for monitoring

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step(loss)

                end_time = time.time()
                iter_time = end_time - start_time
                print(f"Iter time: {iter_time:.3f}s, select time: {select_time:.3f}s, train time: {iter_time - select_time:.3f}s")
                if iter % 10 == 0:
                    wandb.log({
                        "epoch": iter,
                        "MSE Loss": d_loss.item(),
                        "Eikonal Loss": eikonal_loss.item(),
                        "Tension Loss": tension_loss.item(),
                        "Gradient Loss": gradient_loss.item(),
                        "Total Loss": loss.item(),
                        "Learning Rate": optimizer.param_groups[0]['lr'],
                    })
                    # print("d_grad_pred shape:", d_grad_pred.shape) 
                    # print("d_grad_pred_norm", d_grad_pred.norm(2, dim=-1))
                    print(f"Epoch:{iter}\tMSE Loss: {d_loss.item():.3f}\tEikonal Loss: {eikonal_loss.item():.3f}\tTension Loss: {tension_loss.item():.3f}\tGradient Loss: {gradient_loss.item():.3f}\tTotal loss:{loss.item():.3f}\tTime: {time.strftime('%H:%M:%S', time.gmtime())}")
                    model_dict[iter] = model.state_dict()
                    
                    torch.save(model_dict, os.path.join(CUR_PATH,'model_dict.pt'))
                    wandb.save(os.path.join(CUR_PATH,'model_dict.pt'))
                if iter % 1000 == 0 :
                    artifact = wandb.Artifact(name='model_dict', type='model')
                    artifact.add_file(os.path.join(CUR_PATH,'model_dict.pt'))
                    wandb.log_artifact(artifact)
        wandb.finish()
        return model
    
    def inference(self,x,q,model):
        model.eval()
        x,q = x.to(self.device),q.to(self.device)
        # q.requires_grad = True
        x_cat = x.unsqueeze(1).expand(-1,len(q),-1).reshape(-1,3)
        q_cat = q.unsqueeze(0).expand(len(x),-1,-1).reshape(-1,9)
        inputs = torch.cat([x_cat,q_cat],dim=-1)
        cdf_pred = model.forward(inputs)
        return cdf_pred
    
    def inference_d_wrt_q(self,x,q,model,return_grad = True):
        cdf_pred = self.inference(x,q,model)
        d = cdf_pred.reshape(len(x),len(q)).min(dim=0)[0]
        if return_grad:
            grad = torch.autograd.grad(d,q,torch.ones_like(d),retain_graph=True,create_graph=True)[0]
            # dgrad = torch.autograd.grad(grad,q,torch.ones_like(grad),retain_graph=True,create_graph=True)[0]
            return d,grad
        else:
            return d
    
    def inference_d_wrt_x(self, x, q, model, return_grad=True):
        cdf_pred = self.inference(x, q, model)
        d = cdf_pred.reshape(len(x), len(q)).min(dim=0)[0]  # take minimum value

        if return_grad:
            # compute gradient of d w.r.t. x

            grad = torch.autograd.grad(d, x, torch.ones_like(d), retain_graph=True, create_graph=True)[0]

            return d, grad
        else:
            return d
    
    def eval_nn(self,model,num_iter = 3):
        eval_time = False
        eval_acc = True
        if eval_time:
            x = torch.rand(100,3).to(device)-torch.tensor([[0.5,0.5,0]]).to(device)
            q = self.sample_q(batch_q=100)
            time_cost_list = []
            for i in range(100):
                t0 = time.time()
                d = self.inference_d_wrt_q(x,q,model,return_grad = False)
                t1 = time.time()
                grad = torch.autograd.grad(d,q,torch.ones_like(d),retain_graph=True,create_graph=True)[0]
                q_proj = self.projection(q,d,grad)
                t2 = time.time()
                if i >0:
                    time_cost_list.append([t1-t0,t2-t1])
            mean_time_cost = np.mean(time_cost_list,axis=0)
            print(f'inference time cost:{mean_time_cost[0]}\t projection time cost: {mean_time_cost[1]}')

        if eval_acc:
            # bp_sdf model
            bp_sdf = bf_sdf.BPSDF(8,-1.0,1.0,self.panda,device)
            bp_sdf_model = torch.load(os.path.join(CUR_PATH,'./resource/models/BP_8.pt'))

            res = []
            for i in range (1000):
                x = torch.rand(1,3).to(device)-torch.tensor([[0.5,0.5,0]]).to(device)
                q = self.sample_q(batch_q=1000)
                for _ in range (num_iter):
                    d,grad = self.inference_d_wrt_q(x,q,model,return_grad = True)
                    q = self.projection(q,d,grad)
                
                q,grad = q.detach(),grad.detach()   # release memory
                pose = torch.eye(4).unsqueeze(0).expand(len(q),-1,-1).to(self.device).float()
                sdf,_ = bp_sdf.get_whole_body_sdf_batch_xy(x, pose, q[:,2:], q[:,:2], bp_sdf_model,use_derivative=False)

                error = sdf.reshape(-1).abs()
                MAE = error.mean()
                RMSE = torch.sqrt(torch.mean(error**2))
                SR = (error<0.03).sum().item()/len(error)
                res.append([MAE.item(),RMSE.item(),SR])
                print(f'iter {i} finished, MAE:{MAE}\tRMSE:{RMSE}\tSR:{SR}')
            res = np.array(res)
            print(f'MAE:{res[:,0].mean()}\tRMSE:{res[:,1].mean()}\tSR:{res[:,2].mean()}')
            print(f'MAE:{res[:,0].std()}\tRMSE:{res[:,1].std()}\tSR:{res[:,2].std()}')

    def eval_nn_noise(self,model,num_iter = 3):
            bp_sdf = bf_sdf.BPSDF(8,-1.0,1.0,self.panda,device)
            bp_sdf_model = torch.load(os.path.join(CUR_PATH,'./resource/models/BP_8.pt'))

            res = []
            for i in range (1000):
                x = torch.rand(1,3).to(device)-torch.tensor([[0.5,0.5,0]]).to(device)
                noise = torch.normal(0,0.03,(1,3)).to(device)
                x_noise = x + noise
                q = self.sample_q(batch_q=1000)
                for _ in range (num_iter):
                    d,grad = self.inference_d_wrt_q(x_noise,q,model,return_grad = True)
                    q = self.projection(q,d,grad)
                q,grad = q.detach(),grad.detach()   # release memory
                pose = torch.eye(4).unsqueeze(0).expand(len(q),-1,-1).to(self.device).float()
                sdf,_ = bp_sdf.get_whole_body_sdf_batch_xy(x, pose,q[:,2:], q[:,:2], bp_sdf_model,use_derivative=False)
                
                error = sdf.reshape(-1).abs()
                MAE = error.mean()
                RMSE = torch.sqrt(torch.mean(error**2))
                SR = (error<0.03).sum().item()/len(error)
                res.append([MAE.item(),RMSE.item(),SR])
                print(f'iter {i} finished, MAE:{MAE}\tRMSE:{RMSE}\tSR:{SR}')
            res = np.array(res)
            print(f'MAE:{res[:,0].mean()}\tRMSE:{res[:,1].mean()}\tSR:{res[:,2].mean()}')
            print(f'MAE:{res[:,0].std()}\tRMSE:{res[:,1].std()}\tSR:{res[:,2].std()}')

    def check_data(self):
        # x_batch:(batch_x,3)
        # q_batch:(batch_q,7)
        # d:(batch_x,batch_q)
        # grad:(batch_x,batch_q,7)


        x_batch,q_batch,d,grad = self.select_data()
        
        print("d",d, "grad",grad)
        q_proj = self.projection(q_batch,d,grad)
        # find d < 0
        negative_indices = torch.where(d < 0)
        print(f"Negative distances found at indices: {negative_indices}")
        

        # visualize
        import trimesh
        pose = torch.eye(4).unsqueeze(0).to(self.device).float()
        for i, (q0, q1) in enumerate(zip(q_batch, q_proj[0])):
            scene = trimesh.Scene()
            print("q0",q0)
            print("q1",q1)
            # Only show the i-th point
            x_point = x_batch[0].detach().cpu().numpy().reshape(1, 3)
            scene.add_geometry(trimesh.PointCloud(x_point, colors=[255, 0, 0]))

            # Robot at current q0
            robot_mesh0 = self.panda.get_forward_robot_mesh_xy(pose, q0.unsqueeze(0))
            robot_mesh0 = np.sum(robot_mesh0)
            robot_mesh0.visual.face_colors = [0, 255, 0, 100]
            scene.add_geometry(robot_mesh0)

            # Robot at current q1
            robot_mesh1 = self.panda.get_forward_robot_mesh_xy(pose, q1.unsqueeze(0))
            robot_mesh1 = np.sum(robot_mesh1)
            robot_mesh1.visual.face_colors = [0, 0, 255, 100]
            scene.add_geometry(robot_mesh1)
            print(f"q0: {q0}, q1: {q1}")
            scene.show()
        # for i in range(len(negative_indices[0])):
        #     idx = negative_indices[0][i]
        #     idx_q = negative_indices[1][i]
        #     scene = trimesh.Scene()
        #     x_point = x_batch[idx].detach().cpu().numpy().reshape(1, 3)
        #     scene.add_geometry(trimesh.PointCloud(x_point, colors=[255, 0, 0]))

        #     # Robot at current q_batch
        #     robot_mesh0 = self.panda.get_forward_robot_mesh_xy(pose, q_batch[idx_q].unsqueeze(0))
        #     robot_mesh0 = np.sum(robot_mesh0)
        #     robot_mesh0.visual.face_colors = [0, 255, 0, 100]
        #     scene.add_geometry(robot_mesh0)

        #     # Robot at current q_proj
        #     robot_mesh1 = self.panda.get_forward_robot_mesh_xy(pose, q_proj[idx,idx_q].unsqueeze(0))
        #     robot_mesh1 = np.sum(robot_mesh1)
        #     robot_mesh1.visual.face_colors = [0, 0, 255, 100]
        #     scene.add_geometry(robot_mesh1)
        #     print(f"q_batch: {q_batch[idx_q]}, q_proj: {q_proj[idx,idx_q]}")
        #     print("d:", d[idx, idx_q], "grad:", grad[idx, idx_q])
        #     scene.show()

    def check_model(self,model):
      
        x = torch.tensor([[0.25, 0.1, 1.2]], requires_grad=True).to(self.device)
        q = self.sample_q(batch_q=10)

        import trimesh
        pose = torch.eye(4).unsqueeze(0).to(self.device).float()
        q_proj = q
        for _ in range (3):
                    d,grad = self.inference_d_wrt_q(x,q_proj,model,return_grad = True)
                    # print(f"d: {d}, grad: {grad}")
                    d,grad_x = self.inference_d_wrt_x(x,q_proj,model,return_grad = True)
                    # print(f"d_x: {d}, grad_x: {grad_x}")
                    # print(f"q_proj: {q_proj}")
                    q_proj = self.projection(q_proj,d,grad)

                    d,grad = self.inference_d_wrt_q(x,q_proj,model,return_grad = True)
                    print(f"d after projection: {d}, grad after projection: {grad}")

        for (q0, q1) in zip(q, q_proj):
            scene = trimesh.Scene()
            scene.add_geometry(trimesh.PointCloud(x.detach().cpu().numpy().reshape(1, 3), colors=[255, 0, 0]))

             # Robot at current q0
            robot_mesh0 = self.panda.get_forward_robot_mesh_xy(pose, q0.unsqueeze(0))
            robot_mesh0 = np.sum(robot_mesh0)
            robot_mesh0.visual.face_colors = [0, 255, 0, 100]
            scene.add_geometry(robot_mesh0)

            # Robot at current q1
            robot_mesh1 = self.panda.get_forward_robot_mesh_xy(pose, q1.unsqueeze(0))
            robot_mesh1 = np.sum(robot_mesh1)
            robot_mesh1.visual.face_colors = [0, 0, 255, 100]
            scene.add_geometry(robot_mesh1)
            print(f"q0: {q0}, q1: {q1}")
            scene.show()

            
            


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    cdf = CDF(device)
    # cdf.check_data()
    cdf.train_nn(epoches=20000)

    model = MLPRegression(input_dims=12, output_dims=1, mlp_layers=[1024, 1024, 512, 512, 256, 256, 128, 128],skips=[], act_fn=torch.nn.GELU, nerf=True)
    model.load_state_dict(torch.load(os.path.join(CUR_PATH, 'model_dict.pt'), map_location='cuda:0')[19900])
    torch.save(model, os.path.join(CUR_PATH, 'model_gcdf.pth'))
    
    # model = MLPRegression(input_dims=12, output_dims=1, mlp_layers=cdf.mlp_layers,skips=[], act_fn=torch.nn.GELU, nerf=True)
    # # model.load_state_dict(torch.load(os.path.join(CUR_PATH,'model_dict.pt'))[19900])
    # model.load_state_dict(torch.load(os.path.join(CUR_PATH,'model_dict.pt'))[14990])
    # model.to(device)
    # cdf.eval_nn(model)
    # cdf.check_model(model)