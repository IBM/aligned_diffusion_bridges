import traceback
import torch
from torch_geometric.loader import DataLoader
import numpy as np

from sbalign.data import ListDataset
from sbalign.utils.ops import to_numpy
from sbalign.utils.sb_utils import get_t_schedule
from sbalign.utils.sampling import sampling
from sbalign.utils.definitions import DEVICE

from proteins.docking.dock_engine import DockingEngine
from proteins.conf.conf_engine import ConfEngine


class ProgressMonitor:

    def __init__(self, metric_names=None):
        if metric_names is not None:
            self.metric_names = metric_names
            self.metrics = {metric: 0.0 for metric in self.metric_names}
        self.count = 0

    def add(self, metric_dict, batch_size: int = None):
        if not hasattr(self, 'metric_names'):
            self.metric_names = list(metric_dict.keys())
            self.metrics = {metric: 0.0 for metric in self.metric_names}

        self.count += (1 if batch_size is None else batch_size)

        for metric_name, metric_value in metric_dict.items():
            if metric_name not in metric_dict:
                self.metrics[metric_name] = 0.0
                self.metric_names.append(metric_name)
            
            self.metrics[metric_name] += metric_value * (1 if batch_size is None else batch_size)

    def summarize(self):
        return {k: np.round(v / self.count, 4) for k, v in self.metrics.items()}


def train_epoch_sbalign(
        model, loader, 
        optimizer, loss_fn,
        grad_clip_value: float = None, 
        ema_weights=None):

    model.train()
    monitor = ProgressMonitor()

    for idx, data in enumerate(loader):
        optimizer.zero_grad()

        try:
            data = data.to(DEVICE)
            drift_x, doobs_score_x, doobs_score_x_T = model(data)

            loss, loss_dict = loss_fn(drift_x_pred=drift_x,
                                      doobs_score_x_pred=doobs_score_x,
                                      doobs_score_xT_pred=doobs_score_x_T,
                                      data=data)
                                    
            monitor.add(loss_dict)

            loss.backward()

            if grad_clip_value is not None:
                grad_clip_value = 10.0
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            optimizer.step()
            
            if ema_weights is not None:
                ema_weights.update(model.parameters())
            
        except Exception as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                print(e)
                traceback.print_exc()
                continue

    return monitor.summarize()


def test_epoch_sbalign(model, loader, loss_fn):
    model.eval()
    monitor = ProgressMonitor()

    for idx, data in enumerate(loader):
        try:
            with torch.no_grad():
                data = data.to(DEVICE)
                drift_x, doobs_score_x, doobs_score_x_T = model(data)
                
                _, loss_dict = loss_fn(drift_x_pred=drift_x,
                                       doobs_score_x_pred=doobs_score_x,
                                       doobs_score_xT_pred=doobs_score_x_T,
                                       data=data)
                
                monitor.add(loss_dict)

        except Exception as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                print(e)
                continue

    return monitor.summarize()


# -------------------- Inference ------------------------

def inference_epoch_sbalign(model, g, dataset, inference_steps: int = 100, t_max: float =1.0):
    t_schedule = torch.from_numpy(get_t_schedule(inference_steps, t_max=t_max)).float().to(DEVICE)

    pos_T_preds = []
    pos_T = to_numpy(dataset['final'])

    n_samples = len(dataset['initial'])

    input_data = dataset['initial'].to(DEVICE)

    for idx in range(n_samples):
        pos_0 = input_data[idx: idx+1]
        assert pos_0.shape[0] == 1

        pos_T_pred = sampling(pos_0, model, g, inference_steps, t_schedule)
        pos_T_preds.append(to_numpy(pos_T_pred))
    
    pos_T_preds = np.asarray(pos_T_preds).reshape(n_samples, -1)

    rmsd = np.sqrt( ((pos_T_preds - pos_T)**2).sum(axis=1).mean(axis=0) )
    return {'rmsd': np.round(rmsd, 4)}


def inference_epoch_docking(model, g, orig_dataset, num_inference_complexes: int = 10,
                            inference_steps: int = 100, 
                            samples_per_complex: int = 10, rot_vec=None, tr_vec=None):
    
    dataset = ListDataset(
        processed_dir=orig_dataset.full_processed_dir, 
        id_list=orig_dataset.complexes_split[:num_inference_complexes]
    ) 

    engine = DockingEngine(
        model=model, g_fn=g, 
        samples_per_complex=samples_per_complex,
        inference_steps=inference_steps,
        rot_vec=rot_vec, tr_vec=tr_vec
    )

    monitor = ProgressMonitor()
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    traj_dict = {}

    for data in loader:
        trajectory, complex_metrics = engine.dock(data)
        monitor.add(complex_metrics)
        if 'complex_id' in data:
            if len(data['complex_id']) > 0:
                traj_dict[data['complex_id'][0]] = trajectory
    
    return traj_dict, monitor.summarize()


def inference_epoch_conf(model, g, orig_dataset, inference_steps: int = 100,
                         num_inference_proteins: int = 10,
                         samples_per_protein: int = 10):
    
    dataset = ListDataset(
        processed_dir=orig_dataset.full_processed_dir, 
        id_list=orig_dataset.conf_pairs_split[:num_inference_proteins]
    )                     
    
    engine = ConfEngine(
        model=model, g_fn=g, 
        samples_per_protein=samples_per_protein,
        inference_steps=inference_steps
    )

    monitor = ProgressMonitor()
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    traj_dict = {}
    for data in loader:
        trajectory, metrics = engine.generate_conformations(data)
        monitor.add(metrics)

        if "conf_id" in data:
            if len(data['conf_id']) > 0:
                traj_dict[data['conf_id'][0]] = trajectory

    return traj_dict, monitor.summarize()
