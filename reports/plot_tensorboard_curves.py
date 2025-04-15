import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

def f(s):
    """Extracts loss/metric name x from val/x"""
    return s.split("/")[1]

def plot_tensorboard_metrics(logdir, tag_filter="val/", figsize=(12, 14), steps_per_epoch=None):
    """
    Plot selected scalar metrics from a TensorBoard log directory.
    
    Args:
        logdir (str): Path to TensorBoard log directory (e.g. "lightning_logs/version_0").
        tag_filter (str): Substring to filter which tags to plot (e.g., "val/" for validation metrics).
        figsize (tuple): Size of the matplotlib figure.
        steps_per_epoch (int, optional): Number of steps per epoch for x-axis scaling.
    """
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()

    tags = ea.Tags()["scalars"]
    selected_tags = [tag for tag in tags if tag_filter in tag]

    if not selected_tags:
        print(f"No tags matching '{tag_filter}' found in {logdir}.")
        return

    loss_group_1 = []
    loss_group_2 = []
    loss_eval = []
    top_prob_tags = []
    percentage_tags = []
    move_accuracy_tags = []
    lowprob_tags = []
    true_prob_tags = []
    eval_tags = []

    for tag in selected_tags:
        if "loss_total" in tag or "loss_move" in tag:
            loss_group_1.append(tag)
        elif "loss_eval" in tag:
            loss_eval.append(tag)
        elif "loss" in tag:
            loss_group_2.append(tag)
        elif any(k in tag for k in ["top1_prob", "top3_prob", "top5_prob"]):
            top_prob_tags.append(tag)
        elif any(k in tag for k in ["top3_acc", "top5_acc", "move_accuracy"]):
            move_accuracy_tags.append(tag)
        elif "true_prob" in tag:
            true_prob_tags.append(tag)
        elif "lowprob_frac" in tag:
            lowprob_tags.append(tag)
        elif "eval_accuracy" in tag:
            eval_tags.append(tag)

    fig, axs = plt.subplots(nrows=8, figsize=figsize)

    for tag in loss_group_1:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        steps_raw = [e.step for e in events]
        if steps_per_epoch:
            steps = [int(s // steps_per_epoch) for s in steps_raw]
        else:
            steps = steps_raw
        axs[0].plot(steps, values, label=f(tag))
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Total and Move Loss")
    axs[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    axs[0].grid(True)

    for tag in loss_eval:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        steps_raw = [e.step for e in events]
        if steps_per_epoch:
            steps = [int(s // steps_per_epoch) for s in steps_raw]
        else:
            steps = steps_raw
        axs[1].plot(steps, values, label=f(tag))
    axs[1].set_ylabel("Loss")
    axs[1].set_title("Eval Loss")
    axs[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    axs[1].grid(True)

    for tag in loss_group_2:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        steps_raw = [e.step for e in events]
        if steps_per_epoch:
            steps = [int(s // steps_per_epoch) for s in steps_raw]
        else:
            steps = steps_raw
        axs[2].plot(steps, values, label=f(tag))
    axs[2].set_ylabel("Loss")
    axs[2].set_title("Eval, Incheck, and Threat Losses")
    axs[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    axs[2].grid(True)

    for tag in top_prob_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value * 100 for e in events]
        steps_raw = [e.step for e in events]
        if steps_per_epoch:
            steps = [int(s // steps_per_epoch) for s in steps_raw]
        else:
            steps = steps_raw
        axs[3].plot(steps, values, label=f(tag))
    axs[3].set_ylabel("Probability (%)")
    axs[3].set_title("Top-k Probabilities")
    axs[3].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    axs[3].grid(True)

    for tag in move_accuracy_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        steps_raw = [e.step for e in events]
        if steps_per_epoch:
            steps = [int(s // steps_per_epoch) for s in steps_raw]
        else:
            steps = steps_raw
        axs[4].plot(steps, values, label=f(tag))
    axs[4].set_ylabel("Accuracy (%)")
    axs[4].set_title("Move Accuracies")
    axs[4].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    axs[4].grid(True)

    for tag in true_prob_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value * 100 for e in events]
        steps_raw = [e.step for e in events]
        if steps_per_epoch:
            steps = [int(s // steps_per_epoch) for s in steps_raw]
        else:
            steps = steps_raw
        axs[5].plot(steps, values, label=f(tag))
    axs[5].set_ylabel("True Prob (%)")
    axs[5].set_title("Probability of True Move")
    axs[5].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    axs[5].grid(True)

    for tag in lowprob_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value * 100 for e in events]
        steps_raw = [e.step for e in events]
        if steps_per_epoch:
            steps = [int(s // steps_per_epoch) for s in steps_raw]
        else:
            steps = steps_raw
        axs[6].plot(steps, values, label=f(tag))
    axs[6].set_ylabel("Lowprob Fraction (%)")
    axs[6].set_title("Low Probability Fraction (moves < 1%)")
    axs[6].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    axs[6].grid(True)

    for tag in eval_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value * 100 for e in events]
        steps_raw = [e.step for e in events]
        if steps_per_epoch:
            steps = [int(s // steps_per_epoch) for s in steps_raw]
        else:
            steps = steps_raw
        axs[7].plot(steps, values, label=f(tag))
    axs[7].set_ylabel("Accuracy (%)")
    axs[7].set_title("Eval Accuracy")
    axs[7].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    axs[7].grid(True)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()