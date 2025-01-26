from contextlib import nullcontext
import copy
from pathlib import Path
import time
import yaml

import torch
import torch.distributed
import wandb

import numpy as np

from logger.logger import DynamicsLogger
from logger.global_watcher import (
    dump_and_reset,
    enable_logging,
    disable_logging,
    mark_step_end,
)
from optim.weight_averaging import (
    WeightAverager,
    eval_ema,
    eval_wa,
    ExponentialWeightAverager,
)
from optim.grad_checker import (
    direction_cos,
    direction_sub,
)
from .utils import (
    eval,
    get_batch,
    load_checkpoint,
    load_worker_state,
    save_checkpoint,
    save_worker_state,
)
from .linear_probe import train_score_linear_probe


def train(
    model,
    opt,
    datareaders,
    scheduler,
    exp_dir,
    distributed_backend,
    cfg,
    param_name_mapping,
):
    if cfg.compile:
        print(f"Compiling model ...")
        non_compiled_model = model
        model = torch.compile(model)

    if "cuda" in cfg.device:
        type_ctx = torch.amp.autocast(
            device_type="cuda",
            dtype={
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[cfg.dtype],
        )
    else:
        type_ctx = nullcontext()

    weight_averager = None
    if cfg.weight_average:
        # This does generally not support resuming training, but will work if
        # cfg.wa_interval perfectly divides the iteration number of the chkpt.
        # Otherwise, the first avg will not be correctly computed, with a bias
        # towards the first sample and missing values for earlier iterations.
        weight_averager = WeightAverager(
            non_compiled_model,
            horizon=cfg.wa_horizon,
            interval=cfg.wa_interval,
            device=cfg.device,
            save_dir=None if cfg.wa_use_temp_dir else str(Path(exp_dir) / "avgs"),
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
        )

    ema = None
    if cfg.exponential_moving_average:
        ema = ExponentialWeightAverager(
            non_compiled_model,
            interval=cfg.ema_interval,
            decay=cfg.ema_decay,
            device=cfg.device,
            warmup=(cfg.ema_warmup_override or cfg.warmup_steps) if cfg.ema_after_warmup else 0,
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
        )

    if cfg.resume_from:
        # This is a full resume including the model weights, optimizer, state
        # dataloader state, random seed, etc. Not indended for fine tuning or
        # other scenarios where some of these should change.
        print(f"\nResuming Training From {cfg.resume_from}")
        ckpt_dir = Path(cfg.resume_from)
        curr_iter = load_checkpoint(
            model=model,
            opt=opt,
            scheduler=scheduler,
            ckpt_path=ckpt_dir / "main.pt",
            device=cfg.device,
            weight_averager=weight_averager,
            ema=ema,
            resume_from_ema=cfg.resume_from_ema,
            reset_optimizer=cfg.reset_optimizer
        )
        if not curr_iter >= cfg.iterations:
            # only eval run will fire, no need to restore states
            load_worker_state(ckpt_dir)
    else:
        curr_iter = 0

    if distributed_backend.is_master_process() and cfg.log_dynamics:
        with open(cfg.dynamics_logger_cfg, "r") as f:
            dlcfg = yaml.safe_load(f)

        # Hooks into optimizer
        dlogger = DynamicsLogger(
            model, opt, dlcfg, cfg.results_base_folder, wandb=cfg.wandb
        )
        dlogger.iteration = curr_iter

    substep = curr_iter * cfg.acc_steps
    train_reader, val_reader = datareaders["train"], datareaders["val"]
    train_reader.set_step(substep)

    if cfg.shuffle_next_steps:
        train_reader.shuffle_next_steps(
            steps=cfg.shuffle_next_steps,
            seed=cfg.shuffle_next_steps_seed,
            replicate=cfg.shuffle_next_steps_replicate,
            reuse_before=cfg.shuffle_next_steps_use_all_before
        )

    alignment_direction = None
    if cfg.alignment_direction_file:
        alignment_direction = torch.load(cfg.alignment_direction_file, map_location="cuda")

    stats = {"train_loss": [], "val_loss": [], "val_pp": [], "val_acc": []}
    model.train()

    if cfg.one_step is not None:
        cfg.iterations = curr_iter + cfg.one_step

    while curr_iter <= cfg.iterations:
        # Save permanent checkpoint
        if cfg.permanent_ckpt_interval > 0 or curr_iter == cfg.iterations:
            if curr_iter % cfg.permanent_ckpt_interval == 0 or curr_iter == cfg.iterations:
                ckpt_dir = Path(exp_dir) / "ckpts" / str(curr_iter)
                if distributed_backend.is_master_process():
                    save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir, weight_averager=weight_averager, ema=ema)
                ckpt_dir = Path(exp_dir) / "ckpts" / str(curr_iter)
                save_worker_state(ckpt_dir)

        # Save temporary checkpoint for resuming training
        if cfg.latest_ckpt_interval > 0:
            if curr_iter % cfg.latest_ckpt_interval == 0 or curr_iter == cfg.iterations:
                ckpt_dir = Path(exp_dir) / "ckpts" / "latest"
                if distributed_backend.is_master_process():
                    save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir, weight_averager=weight_averager, ema=ema)
                ckpt_dir = Path(exp_dir) / "ckpts" / "latest"
                save_worker_state(ckpt_dir)

        ws = distributed_backend.get_world_size()
        tokens = ws * substep * cfg.sequence_length * cfg.batch_size
        epoch = tokens / train_reader.num_tokens
        if (
            curr_iter % cfg.eval_interval == 0
            or curr_iter == cfg.iterations
            or (curr_iter in cfg.full_eval_at)
        ):
            eval_and_log(
                curr_iter,
                epoch,
                model,
                val_reader,
                type_ctx,
                distributed_backend,
                cfg,
                opt,
                full_eval=(curr_iter in cfg.full_eval_at),
            )
            if cfg.probe_states:
                final_probe_train_loss, probes_loss = train_score_linear_probe(
                    model, train_reader, cfg.device
                )
                stats = {}
                stats['probe_train_ppl'] = 2.71828 ** final_probe_train_loss
                stats |= {
                    f"probe_eval_ppl/layer_{k}": 2.71828 ** v for k, v in probes_loss.items()
                }
                outputs = [dict()] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(outputs, stats)
                if cfg.wandb and distributed_backend.is_master_process():
                    wandb.log(
                        {
                            "iter": curr_iter,
                        } | {
                            k: np.mean([i[k] for i in outputs])
                            for k in outputs[0]
                        }
                    )

            if curr_iter > cfg.wa_interval and cfg.weight_average:
                eval_wa(
                    curr_iter,
                    non_compiled_model,
                    weight_averager,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )
            if cfg.exponential_moving_average:
                eval_ema(
                    curr_iter,
                    non_compiled_model,
                    ema,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )

        if curr_iter == cfg.iterations:
            # Save checkpoints and evaluate at final iteration, but no need to train further
            break

        # Train model
        t_start = time.perf_counter_ns()
        batches = [get_batch(train_reader, device=cfg.device) for _ in range(cfg.acc_steps)]

        enable_logging()
        for microstep_idx in range(cfg.acc_steps):  # gradient accumulation
            x, y = batches[microstep_idx]
            with distributed_backend.get_context_for_microstep_forward(
                    model=model,
                    microstep_idx=microstep_idx,
                    gradient_accumulation_steps=cfg.acc_steps,
                ):
                with type_ctx:
                    outputs = model(x, targets=y)

                loss = outputs["loss"] / cfg.acc_steps
                loss.backward()
                substep += 1
                mark_step_end()
        disable_logging()

        grad_norm = None
        if cfg.grad_clip != 0.0 and cfg.opt != "SLS":
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        if cfg.opt == "SFAdamW":
            opt.train()

        if alignment_direction is not None:
            state_before = {k: v.detach().clone() for k, v in model.state_dict().items()}

        will_log = (
            cfg.log_interval
            and curr_iter % cfg.log_interval == 0
            and distributed_backend.is_master_process()  # Only log on master rank
        )

        grads_alignment, momentum = None, None
        if will_log:
            def extract_momentum_states(optimizer):
                momentum_states = {}
                for param_group in optimizer.param_groups:
                    for param in param_group['params']:
                        if param in optimizer.state:
                            state = optimizer.state[param]
                            exp_avg = state.get('exp_avg', None)
                            exp_avg_sq = state.get('exp_avg_sq', None)
                            momentum_states[param] = {
                                'exp_avg': exp_avg,
                                'exp_avg_sq': exp_avg_sq
                            }
                return momentum_states

            momentum = extract_momentum_states(opt)
            # calculating cos before step
            cosines = {
                k: direction_cos(
                    {k: momentum[p]['exp_avg']},
                    {k: p.grad},
                )
                for k, p in model.named_parameters() if p.grad is not None
            }
            cosines_values = list(cosines.values())
            grads_alignment = dict(
                grads_momentum_alignment=np.mean(cosines_values),
                grads_momentum_alignment_median=np.median(cosines_values),
                **{
                    f"grads_momentum_alignment_q{q}": np.quantile(cosines_values, q / 100)
                    for q in range(0, 101, 10)
                },
                **{
                    f"grads_momentum_alignment_param/{k}": v
                    for k, v in cosines.items()
                }
            )


        if cfg.opt == "SLS":
            @torch.no_grad()
            def get_loss():
                loss = 0
                for (x, y) in batches:
                    outputs = model(x, targets=y)
                    loss = loss + outputs["loss"]
                return loss / cfg.acc_steps

            model.eval()  # we don't want dropout to add noise to the line search
            loss = opt.step(get_loss)
            model.train()
        else:
            opt.step()

        if scheduler is not None:
            scheduler.step()
        if cfg.weight_average:
            weight_averager.step(non_compiled_model, distributed_backend.is_master_process())
        if cfg.exponential_moving_average:
            ema.step(non_compiled_model, distributed_backend.is_master_process())
        dt = (time.perf_counter_ns() - t_start) / 1e9

        curr_iter += 1

        if will_log:
            train_loss = loss.detach().cpu().item() * cfg.acc_steps

            if cfg.opt == "SLS":
                current_lrs = [opt.params["step_size"]]
            else:
                current_lrs = [param_group["lr"] for param_group in opt.param_groups]

            print(
                f"Train: Iter={curr_iter} ({epoch:0.3f} epochs) "
                f"train_loss={train_loss:.3f} iter_dt={dt:.2e}s "
                f"lr={current_lrs[0]:.2e}"
            )

            stats = dump_and_reset()
            stats |= grads_alignment
            stats['grad_norm'] = grad_norm


            if alignment_direction is not None:
                stats["direction_gradients_cos"] = direction_cos(
                    model.state_dict(),
                    alignment_direction,
                )

                stats["direction_update_cos"] = direction_cos(
                    direction_sub(model.state_dict(), state_before),
                    alignment_direction,
                )

                if cfg.opt == "adamw":
                    stats["direction_momentum_cos"] = direction_cos(
                        {k: momentum[p]['exp_avg'] for k, p in model.named_parameters()},
                        alignment_direction,
                    )

            if cfg.wandb:
                wandb.log(
                    {
                        "iter": curr_iter,
                        "train/loss": train_loss,
                        "train/perplexity": 2.71828**train_loss,
                        "lr": current_lrs[0],
                        "iter_dt": dt,
                    } | stats
                )

        opt.zero_grad(set_to_none=True)

    return stats


@torch.no_grad()
def eval_and_log(
    curr_iter,
    epoch,
    model,
    val_reader,
    type_ctx,
    distributed_backend,
    cfg,
    opt,
    full_eval=False,
):
    if not distributed_backend.is_master_process():
        # Only evaluate and log on master rank
        return

    model.eval()
    if cfg.opt == "SFAdamW":
        opt.eval()

    if curr_iter == cfg.iterations or full_eval:
        max_num_batches = val_reader.num_batches()
    else:
        max_num_batches = cfg.eval_batches

    # to make sure we start from the beginning of the validation set,
    # i.e. repeat the same batches
    val_reader.set_step(0)
    val_acc, val_loss, val_perplexity = eval(
        model,
        val_reader,
        cfg.device,
        max_num_batches=max_num_batches,
        ctx=type_ctx,
        cfg=cfg,
    )

    print(
        f">Eval: Iter={curr_iter} ({epoch:0.3f} epochs) "
        f"val_loss={val_loss:.3f} "
        f"val_pp={val_perplexity:.3f} "
        f"val_acc={val_acc:3f}"
    )

    if cfg.wandb:
        if curr_iter == cfg.iterations or full_eval:
            logs = {
                "iter": curr_iter,
                "final-val/loss": val_loss,
                "final-val/perplexity": val_perplexity,
                "final-val/acc": val_acc,
            }
        else:
            logs = {
                "iter": curr_iter,
                "val/loss": val_loss,
                "val/perplexity": val_perplexity,
                "val/acc": val_acc,
            }

        wandb.log(logs)
        if cfg.eval_seq_prefix != "none" and (
            curr_iter % (cfg.eval_interval * 5) == 0 or curr_iter == cfg.iterations
        ):
            text_table = wandb.Table(columns=["itr", "val-pp", "text"])

            out_str = distributed_backend.get_raw_model(model).generate_from_string(
                cfg.eval_seq_prefix,
                max_new_tokens=40,
                temperature=0.9,
                top_k=None,
            )
            text_table.add_data(curr_iter, val_perplexity, out_str)
            # why a copy? see github.com/wandb/wandb/issues/2981
            wandb.log({f"generated-text-{wandb.run.name}": copy.copy(text_table)})
    model.train()
