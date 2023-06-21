import os
import yaml
import torch
import wandb
import numpy as np
import copy
import math

from sbalign.data.datasets import build_data_loader
from sbalign.models.base import AlignedSB
from sbalign.training.epoch_fns import train_epoch_sbalign, test_epoch_sbalign, inference_epoch_sbalign
from sbalign.training.updaters import get_optimizer, get_scheduler, get_ema
from sbalign.training.losses import loss_fn_from_args
from sbalign.utils.helper import count_parameters
from sbalign.utils.setup import wandb_setup, parse_train_args, update_args_from_config
from sbalign.utils.definitions import DEVICE
from sbalign.training.diffusivity import get_diffusivity_schedule


def train(args, train_loader, val_loader, model, optimizer, scheduler, ema_weights=None, log_dir=None):
    best_val_loss = math.inf
    best_val_inference_value = math.inf if args.inference_goal == 'min' else 0
    best_epoch = 0
    best_val_inference_epoch = 0

    g = get_diffusivity_schedule(args.diffusivity_schedule, args.max_diffusivity)
    loss_fn = loss_fn_from_args(args)
    
    logs = {'val_loss': math.inf, "val_inference_rmsd": math.inf}

    for epoch in range(args.n_epochs):
        train_losses = train_epoch_sbalign(
                model=model, 
                loader=train_loader, 
                optimizer=optimizer, 
                loss_fn=loss_fn,
                grad_clip_value=args.grad_clip_value, 
                ema_weights=ema_weights
            )

        print_msg = f"Epoch {epoch+1}: "
        for item, value in train_losses.items():
            if item == "loss":
                print_msg += f"Training Loss: {np.round(value, 4)} "
            else:
                print_msg += f"{item} {np.round(value, 4)} "
        logs.update({'train_' + k: v for k, v in train_losses.items()})
        print(print_msg, flush=True)

        # Load ema parameters into model
        if ema_weights is not None:
            ema_weights.store(model.parameters())
            ema_weights.copy_to(model.parameters())

        # Compute losses on validation set
        val_losses = test_epoch_sbalign(model=model, loader=val_loader, loss_fn=loss_fn)
        print_msg = f"Epoch {epoch+1}: "
        for item, value in val_losses.items():
            if item == "loss":
                print_msg += f"Validation Loss: {np.round(value, 4)} "
            else:
                print_msg += f"{item} {np.round(value, 4)} "
        logs.update({'val_' + k: v for k, v in val_losses.items()})
        print(print_msg, flush=True)
        print(flush=True)

        # Inference on validation set
        if args.inference_every > 0 and (epoch + 1) % args.inference_every == 0:
            inference_metrics = inference_epoch_sbalign(model=model, g=g, dataset=val_loader.dataset.data, 
                                             inference_steps=args.inference_steps, t_max=1.0)
            
            print_msg = f"Epoch {epoch+1}: "
            for item, value in inference_metrics.items():
                print_msg += f"{item} {value}"
            logs.update({'val_inference_' + k: v for k, v in inference_metrics.items()})                         
            print(print_msg, flush=True)

        if ema_weights is not None:
            ema_state_dict = copy.deepcopy(model.state_dict() if DEVICE == 'cuda' else model.state_dict())
            ema_weights.restore(model.parameters())

        # Write logs to wandb
        if args.online:
            logs.update({'train_' + k: v for k, v in train_losses.items()})
            logs.update({'val_' + k: v for k, v in val_losses.items()})
            logs.update({'val_inference_' + k: v for k, v in inference_metrics.items()})
            logs['current_lr'] = optimizer.param_groups[0]['lr']
            wandb.log(logs, step=epoch + 1)

        model_dict = model.state_dict()

        if args.inference_every > 0:
            if args.inference_metric in logs.keys() and \
                    (args.inference_goal == 'min' and logs[args.inference_metric] < best_val_inference_value or
                    args.inference_goal == 'max' and logs[args.inference_metric] > best_val_inference_value):
                best_val_inference_value = logs[args.inference_metric]
                best_val_inference_epoch = epoch

                if log_dir is not None:
                    model_file = os.path.join(log_dir, 'best_inference_epoch_model.pt')
                    ema_file = os.path.join(log_dir, 'best_ema_inference_epoch_model.pt')

                    torch.save(model_dict, model_file)
                    torch.save(ema_state_dict, ema_file)
                    
                    print(f"After best inference, saving model to {model_file}", flush=True)
                    print(f"After best inference, saving ema to {ema_file}", flush=True)
                    print(flush=True)

        if val_losses['loss'] <= best_val_loss:
            best_val_loss = val_losses['loss']
            best_epoch = epoch

            if log_dir is not None:
                model_file = os.path.join(log_dir, "best_model.pt")
                ema_file = os.path.join(log_dir, "best_ema_model.pt")

                torch.save(model_dict, os.path.join(log_dir, 'best_model.pt'))
                torch.save(ema_state_dict, os.path.join(log_dir, 'best_ema_model.pt'))
        
            print(f"After best validation, saving model to {model_file}", flush=True)
            print(f"After best validation, saving ema to {ema_file}", flush=True)
            print(flush=True)

        if scheduler is not None:
            if args.inference_every > 0:
                scheduler.step(logs[args.early_stop_metric])

        if log_dir is not None:
            print(f"Saving last model to {log_dir}/last_model.pt", flush=True)
            torch.save({
                'epoch': epoch,
                'model': model_dict,
                'optimizer': optimizer.state_dict(),
                'ema_weights': ema_weights.state_dict(),
            }, os.path.join(log_dir, 'last_model.pt'))
            print(flush=True)

    print(f"Best Validation Loss {best_val_loss} on Epoch {best_epoch}", flush=True)
    print(f"Best Inference Metric {best_val_inference_value} on Epoch {best_val_inference_epoch}", flush=True)


def main(cmd_args=None):
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=4)

    print(flush=True)
    # Load args from command line and replace values with those from config
    if isinstance(cmd_args, list) or cmd_args is None:
        args = parse_train_args(cmd_args)
        args = update_args_from_config(args)
    elif isinstance(cmd_args, dict):
        args = cmd_args
    else:
        raise ValueError("Cannot parse params")
    
    print(f"Args: {args}", flush=True)
    print(flush=True)

    # Wandb setup
    if args.online:
        wandb_setup(args)

    if args.run_name is None:
        args.run_name = f"dataset={args.dataset}_samples={args.n_samples}_hdim={args.h_dim}_layers={args.n_layers}"

    print(f"Experiment Name: {args.run_name}", flush=True)
    print(flush=True)

    # Datasets
    train_loader, val_loader = build_data_loader(args)
    n_train = len(train_loader.dataset)
    n_valid = len(val_loader.dataset)
    print(f"Num Train: {n_train}, Num Valid: {n_valid}", flush=True)
    print(f"Train loader inspection:", train_loader.dataset.data["initial"].mean(axis=0))
    print(f"Val loader inspection:", val_loader.dataset.data["initial"].mean(axis=0))
    print(flush=True)

    # Model
    model = AlignedSB(timestep_emb_dim=args.timestep_emb_dim,
                     n_layers=args.n_layers, in_dim=args.in_dim, out_dim=args.out_dim,
                     h_dim=args.h_dim, activation=args.activation, 
                     dropout_p=args.dropout_p, use_drift_in_doobs=args.use_drift_in_doobs)

    n_params = count_parameters(model=model, log_to_wandb=False and args.online)
    print(f"Model with {n_params / (10**6)}M parameters", flush=True)
    model.to(DEVICE)
    print(flush=True)

    # Optimizers
    optimizer = get_optimizer(model=model, optim_name=args.optim_name,
                              lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer=optimizer, scheduler_name=args.scheduler,
                              scheduler_mode=args.scheduler_mode)
    ema = get_ema(model=model, decay_rate=args.ema_decay_rate)

    # Recording configuration
    if args.log_dir is not None:
        log_dir = os.path.join(args.log_dir, args.run_name)
        os.makedirs(log_dir, exist_ok=True)
        config_file = os.path.join(log_dir, "config_train.yml")

        yaml_dump = yaml.dump(args.__dict__)   
        with open(config_file, "w") as f:
            f.write(yaml_dump)

        print(f"Saved model config to {config_file}", flush=True)
        print(flush=True)

    else:
        log_dir = None

    print(f"Training model for {args.n_epochs} epochs...", flush=True)
    train(args=args, train_loader=train_loader, 
          val_loader=val_loader, model=model, optimizer=optimizer,
          scheduler=scheduler, ema_weights=ema, log_dir=log_dir)

    return model


if __name__ == "__main__":
    main()
