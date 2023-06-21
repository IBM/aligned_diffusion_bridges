import os
import yaml
import torch
import wandb
import numpy as np
import copy
import math

from proteins.conf.dataset import build_data_loader
from proteins.conf.models import build_model_from_args

from sbalign.training.epoch_fns import train_epoch_sbalign, test_epoch_sbalign, inference_epoch_conf
from sbalign.training.losses import loss_fn_from_args
from sbalign.training.updaters import get_optimizer, get_scheduler, get_ema
from sbalign.utils.sb_utils import get_diffusivity_schedule
from sbalign.utils.helper import count_parameters
from sbalign.utils.setup import wandb_setup, parse_conf_train_args, update_args_from_config
from sbalign.utils.definitions import DEVICE


def train(args, train_loader, val_loader, model, optimizer, scheduler, ema_weights=None, log_dir=None):
    best_val_loss = math.inf
    best_val_inference_value = math.inf if args.inference_goal == 'min' else 0
    best_epoch = 0
    best_val_inference_epoch = 0

    g = get_diffusivity_schedule(args.diffusivity_schedule, args.max_diffusivity)
    loss_fn = loss_fn_from_args(args)

    logs = {'val_loss': math.inf, "val_inference_rmsd": math.inf}

    for epoch in range(args.n_epochs):
        log_dict = {}
        
        train_losses = train_epoch_sbalign(
                model=model, loader=train_loader, 
                optimizer=optimizer, loss_fn=loss_fn,
                grad_clip_value=args.grad_clip_value, 
                ema_weights=ema_weights, 
            )
        
        # Print training metrics
        print_msg = f"Epoch {epoch+1}: "
        for item, value in train_losses.items():
            if item == "loss":
                print_msg += f"Training Loss: {np.round(value, 4)} "
            else:
                print_msg += f"{item}: {np.round(value, 4)} "
        logs.update({'train_' + k: v for k, v in train_losses.items()})
        print(print_msg, flush=True)

        # Load ema parameters into model
        if ema_weights is not None:
            ema_weights.store(model.parameters())
            ema_weights.copy_to(model.parameters())

        # Compute losses on validation set
        val_losses = test_epoch_sbalign(model=model, loader=val_loader, loss_fn=loss_fn)

        # Print validation metrics
        print_msg = f"Epoch {epoch+1}: "
        for item, value in val_losses.items():
            if item == "loss":
                print_msg += f"Validation Loss: {np.round(value, 4)} "
            else:
                print_msg += f"{item}: {np.round(value, 4)} "
        logs.update({'val_' + k: v for k, v in val_losses.items()})
        print(print_msg, flush=True)  

        # Inference on validation set
        if args.inference_every > 0 and (epoch + 1) % args.inference_every == 0:
            traj_dict, inference_metrics = inference_epoch_conf(
                                            model=model, g=g, 
                                            orig_dataset=val_loader.dataset,
                                            num_inference_proteins=args.num_inference_proteins,
                                            inference_steps=args.inference_steps,
                                            samples_per_protein=args.samples_per_protein
                                        )
            
            print_msg = f"Epoch {epoch+1}: Inference "
            for item, value in inference_metrics.items():
                print_msg += f"{item}: {np.round(value, 4)} "                    
            print(print_msg, flush=True)
            logs.update({'val_inference_' + k: v for k, v in inference_metrics.items()})
        print(flush=True)

        if ema_weights is not None:
            ema_state_dict = copy.deepcopy(model.state_dict() if DEVICE == 'cuda' else model.state_dict())
            ema_weights.restore(model.parameters())

        model_dict = model.state_dict()

        if args.inference_every > 0:
            if args.inference_metric in logs.keys() and \
                    (args.inference_goal == 'min' and logs[args.inference_metric] < best_val_inference_value or
                    args.inference_goal == 'max' and logs[args.inference_metric] > best_val_inference_value):
                best_val_inference_value = logs[args.inference_metric]
                best_val_inference_epoch = epoch

                if log_dir is not None:
                    model_file = os.path.join(log_dir, 'best_inference_epoch_model.pt')
                    print(f"After best inference, saving model to {model_file}", flush=True)
                    torch.save(model_dict, model_file)

                    if ema_weights is not None:
                        ema_file = os.path.join(log_dir, 'best_ema_inference_epoch_model.pt')
                        print(f"After best inference, saving ema to {ema_file}", flush=True)
                        torch.save(ema_state_dict, ema_file)
                    
                    print(f"After best inference, saving trajectories to {log_dir}/trajectories", flush=True)
                    os.makedirs(os.path.join(log_dir, "trajectories"), exist_ok=True)
                    for complex_id in traj_dict:
                        trajectory = traj_dict[complex_id]
                        traj_file = f"{log_dir}/trajectories/{complex_id}.npy"
                        np.save(traj_file, trajectory)
                    print(flush=True)

        # Write logs to wandb
        if args.wandb_mode == "online":
            # Logging metrics and losses
            log_dict.update({'train_' + k: v for k, v in train_losses.items()})
            log_dict.update({'val_' + k: v for k, v in val_losses.items()})
            if args.inference_every > 0 and (epoch + 1) % args.inference_every == 0:
                log_dict.update({'val_inference_' + k: v for k, v in inference_metrics.items()})     
            log_dict['current_lr'] = optimizer.param_groups[0]['lr']
            log_dict["step"] = epoch + 1
            wandb.log(log_dict)

        if log_dir is not None:
            model_file = os.path.join(log_dir, "best_model.pt")
            print(f"After best validation, saving model to {model_file}", flush=True)
            torch.save(model_dict, os.path.join(log_dir, 'best_model.pt'))

            if ema_weights is not None:
                ema_file = os.path.join(log_dir, "best_ema_model.pt")
                print(f"After best validation, saving ema to {ema_file}", flush=True)
                torch.save(ema_state_dict, os.path.join(log_dir, 'best_ema_model.pt'))
            print(flush=True)

        if scheduler is not None:
            if args.early_stop_metric in logs:
                scheduler.step(logs[args.early_stop_metric])
            else:
                scheduler.step(logs["val_loss"])

        if log_dir is not None:
            print(f"Saving last model to {log_dir}/last_model.pt", flush=True)
            save_dict = {
                'epoch': epoch,
                'model': model_dict,
                'optimizer': optimizer.state_dict(),
            }

            if ema_weights is not None:
                save_dict['ema_weights'] = ema_weights.state_dict()

            torch.save(save_dict, os.path.join(log_dir, 'last_model.pt'))
            print(flush=True)

    print(f"Best Validation Loss {best_val_loss} on Epoch {best_epoch}", flush=True)
    print(f"Best Inference Metric {best_val_inference_value} on Epoch {best_val_inference_epoch}", flush=True)


def main(cmd_args=None):
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=4)

    # Load args from command line and replace values with those from config
    print(flush=True)
    args = parse_conf_train_args(cmd_args=cmd_args)
    args = update_args_from_config(args=args)

    # Wandb setup
    wandb_setup(args)
    args.wandb_dir = os.path.dirname(wandb.run.dir)

    print(f"Args: {args}", flush=True)
    print(flush=True)

    print(f"Experiment Name: {args.run_name}", flush=True)
    print(flush=True)

    # Datasets
    train_loader, val_loader = build_data_loader(args)

    # Model
    model = build_model_from_args(args)

    n_params = count_parameters(model=model, log_to_wandb=False and args.online)
    print(f"Model with {n_params / (10**6)}M parameters", flush=True)
    model.to(DEVICE)
    print(flush=True)

    # Optimizers
    optimizer = get_optimizer(model=model, optim_name=args.optim_name,
                              lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer=optimizer, scheduler_name=args.scheduler,
                              scheduler_mode=args.scheduler_mode, factor=0.7,
                              patience=args.scheduler_patience, min_lr=args.lr / 100)
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
