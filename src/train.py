import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from utils import compute_mpjpe, compute_pa_mpjpe

from visualize import visualize_comparison, fig_to_image

logger = logging.getLogger(__name__)


def train_model(
    model,
    model_type,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    writer,
    gradient_accumulation_steps=1,
    start_step=0,
    num_steps=None,
    eval_interval_steps=1000,
    checkpoint_prefix="checkpoint",
    preview_interval_steps=50,
):
    """
    Train the model, using validation loader for preview visualizations and full evaluation.
    Periodically save model every eval_interval_steps.
    Logs training loss, validation loss, MPJPE, and PA-MPJPE.
    `global_step` is incremented after `gradient_accumulation_steps` raw batches are processed.
    """

    global_step = start_step
    # Initialize step_in_epoch based on start_step to correctly align accumulation logic when resuming
    step_in_epoch = start_step * gradient_accumulation_steps

    val_preview_iter = iter(val_loader) if val_loader is not None else None

    model.train()
    optimizer.zero_grad()  # Initial zero_grad

    target_steps = num_steps if num_steps is not None else float("inf")

    pbar_total = num_steps if num_steps is not None else None
    pbar = tqdm(
        total=pbar_total, desc="Training Steps", initial=global_step, unit="step"
    )

    running_total_train_loss = 0.0
    num_samples_total_train = 0

    accum_loss_for_log = 0.0
    accum_comp_for_log = {}
    batch_accum_count_for_log = 0

    epoch_counter = 0
    training_complete = False
    while global_step < target_steps and not training_complete:
        epoch_counter += 1
        if hasattr(train_loader.sampler, "set_epoch") and callable(
            train_loader.sampler.set_epoch
        ):
            train_loader.sampler.set_epoch(epoch_counter)

        for _, batch in enumerate(train_loader):
            if global_step >= target_steps:
                training_complete = True
                break

            model.train()  # Ensure model is in training mode

            # Move data to device
            images = batch["image"].to(device, non_blocking=True)
            depths = batch["depth"].to(device, non_blocking=True)
            keypoints_2d = batch["keypoints_2d"].to(device, non_blocking=True)
            joints_3d_gt = batch["joints_3d"].to(device, non_blocking=True)
            current_batch_size = images.size(0)

            # Forward pass
            outputs = model(images, depths, keypoints_2d)

            # Calculate loss
            loss, comps_from_criterion = criterion(outputs, joints_3d_gt)

            # Normalize loss for gradient accumulation
            loss_scaled = loss / gradient_accumulation_steps

            # Backward pass
            loss_scaled.backward()

            # Accumulate loss and components for eventual logging
            batch_loss_unscaled = loss.item()
            accum_loss_for_log += batch_loss_unscaled * current_batch_size
            running_total_train_loss += (
                batch_loss_unscaled * current_batch_size
            )  # For overall average
            num_samples_total_train += current_batch_size

            for k_comp, v_comp_tensor_or_float in comps_from_criterion.items():
                v_comp_float = (
                    v_comp_tensor_or_float.item()
                    if isinstance(v_comp_tensor_or_float, torch.Tensor)
                    else v_comp_tensor_or_float
                )
                accum_comp_for_log[k_comp] = (
                    accum_comp_for_log.get(k_comp, 0.0)
                    + v_comp_float * current_batch_size
                )

            batch_accum_count_for_log += current_batch_size
            step_in_epoch += 1  # Increment for every raw batch processed

            # Optimizer step, global_step increment, and logging
            if step_in_epoch % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1  # Increment global_step after an optimizer step
                pbar.update(1)  # Update progress bar for one effective step

                if batch_accum_count_for_log > 0:
                    avg_loss_for_log_period = (
                        accum_loss_for_log / batch_accum_count_for_log
                    )

                    writer.add_scalar(
                        "Loss/train_step", avg_loss_for_log_period, global_step
                    )

                    for k_comp, v_sum_comp in accum_comp_for_log.items():
                        if v_sum_comp != 0.0 or "total_loss" in k_comp.lower():
                            writer.add_scalar(
                                f"Loss_Components/{k_comp}",
                                v_sum_comp / batch_accum_count_for_log,
                                global_step,
                            )

                # Reset accumulators for the next set of raw batches
                accum_loss_for_log = 0.0
                accum_comp_for_log = {}
                batch_accum_count_for_log = 0

                # Preview on validation set
                if global_step % preview_interval_steps == 0 and val_loader is not None:
                    if val_preview_iter is None:
                        val_preview_iter = iter(val_loader)
                    try:
                        val_batch_preview = next(val_preview_iter)
                    except StopIteration:
                        val_preview_iter = iter(val_loader)
                        val_batch_preview = next(val_preview_iter)

                    model.eval()
                    with torch.no_grad():
                        v_imgs_preview = val_batch_preview["image"].to(
                            device, non_blocking=True
                        )
                        v_depths_preview = val_batch_preview["depth"].to(
                            device, non_blocking=True
                        )
                        v_kp_preview = val_batch_preview["keypoints_2d"].to(
                            device, non_blocking=True
                        )
                        v_joints_gt_preview = val_batch_preview["joints_3d"].to(
                            device, non_blocking=True
                        )
                        preds_preview = model(
                            v_imgs_preview, v_depths_preview, v_kp_preview
                        )
                        p0 = preds_preview[0].cpu().numpy()
                        g0 = v_joints_gt_preview[0].cpu().numpy()
                        img0_for_preview = v_imgs_preview[0].cpu()
                        fig = visualize_comparison(
                            img0_for_preview,
                            p0,
                            g0,
                            title=f"Val Preview Step {global_step}",
                        )
                        comp_img_for_tb = fig_to_image(fig)
                        writer.add_image(
                            "Val_Preview/comparison",
                            np.array(comp_img_for_tb).transpose(2, 0, 1),
                            global_step,
                        )
                        plt.close(fig)
                    model.train()

                # Full evaluation and checkpoint saving
                if global_step % eval_interval_steps == 0:
                    if val_loader is not None:
                        logger.info(
                            f"\nPerforming full validation at step {global_step}..."
                        )
                        model.eval()
                        val_epoch_losses_list = []
                        val_epoch_mpjpes_list = []
                        val_epoch_pa_mpjpes_list = []
                        val_accum_comps_for_log = {}
                        val_samples_count = 0
                        with torch.no_grad():
                            for _, val_batch in enumerate(
                                tqdm(
                                    val_loader,
                                    desc=f"Validation at Step {global_step}",
                                    leave=False,
                                )
                            ):
                                v_eval_images = val_batch["image"].to(
                                    device, non_blocking=True
                                )
                                v_eval_depths = val_batch["depth"].to(
                                    device, non_blocking=True
                                )
                                v_eval_keypoints_2d = val_batch["keypoints_2d"].to(
                                    device, non_blocking=True
                                )
                                v_eval_joints_3d_gt = val_batch["joints_3d"].to(
                                    device, non_blocking=True
                                )
                                v_current_batch_size = v_eval_images.size(0)
                                v_eval_outputs = model(
                                    v_eval_images, v_eval_depths, v_eval_keypoints_2d
                                )
                                v_loss_val, v_comps_from_criterion = criterion(
                                    v_eval_outputs, v_eval_joints_3d_gt
                                )
                                val_epoch_losses_list.append(
                                    v_loss_val.item() * v_current_batch_size
                                )
                                for (
                                    k_comp,
                                    v_comp_tensor_or_float,
                                ) in v_comps_from_criterion.items():
                                    v_comp_float = (
                                        v_comp_tensor_or_float.item()
                                        if isinstance(
                                            v_comp_tensor_or_float, torch.Tensor
                                        )
                                        else v_comp_tensor_or_float
                                    )
                                    val_accum_comps_for_log[k_comp] = (
                                        val_accum_comps_for_log.get(k_comp, 0.0)
                                        + v_comp_float * v_current_batch_size
                                    )
                                pred_joints_for_metric = v_eval_outputs
                                current_mpjpe = compute_mpjpe(
                                    pred_joints_for_metric, v_eval_joints_3d_gt
                                )
                                current_pa_mpjpe = compute_pa_mpjpe(
                                    pred_joints_for_metric, v_eval_joints_3d_gt
                                )
                                val_epoch_mpjpes_list.append(
                                    current_mpjpe.item() * v_current_batch_size
                                )
                                val_epoch_pa_mpjpes_list.append(
                                    current_pa_mpjpe.item() * v_current_batch_size
                                )
                                val_samples_count += v_current_batch_size

                        avg_val_loss = (
                            sum(val_epoch_losses_list) / val_samples_count
                            if val_samples_count > 0
                            else 0.0
                        )
                        avg_mpjpe = (
                            sum(val_epoch_mpjpes_list) / val_samples_count
                            if val_samples_count > 0
                            else 0.0
                        )
                        avg_pa_mpjpe = (
                            sum(val_epoch_pa_mpjpes_list) / val_samples_count
                            if val_samples_count > 0
                            else 0.0
                        )
                        writer.add_scalar(
                            "Loss/validation_epoch_avg", avg_val_loss, global_step
                        )
                        writer.add_scalar(
                            "Metrics/MPJPE_validation_epoch_avg", avg_mpjpe, global_step
                        )
                        writer.add_scalar(
                            "Metrics/PA_MPJPE_validation_epoch_avg",
                            avg_pa_mpjpe,
                            global_step,
                        )
                        for k_comp, v_sum_comp in val_accum_comps_for_log.items():
                            if val_samples_count > 0:
                                writer.add_scalar(
                                    f"Loss_Components_Val/{k_comp}",
                                    v_sum_comp / val_samples_count,
                                    global_step,
                                )
                        logger.info(
                            f"Step {global_step}: Val Loss: {avg_val_loss:.4f}, MPJPE: {avg_mpjpe:.2f} mm, PA-MPJPE: {avg_pa_mpjpe:.2f} mm"
                        )

                    ckpt = {
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "model_args": model.config.to_dict(),
                        "model_type": model_type,
                    }
                    path = f"{checkpoint_prefix}_{model_type}_step_{global_step}.pth"
                    torch.save(ckpt, path)
                    logger.info(f"Saved checkpoint at step {global_step} to {path}")
                    model.train()

                if global_step >= target_steps:
                    training_complete = True
                    break

            if training_complete:
                break

    pbar.close()
    return model, global_step
