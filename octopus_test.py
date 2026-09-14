# -*- coding: utf-8 -*-
from octopus_v7 import GradientOctopus70
import random

def run_analytics():
    print("=" * 65)
    print("  TEST: 20 examples, target = avg(inputs) + 10, some noisy")
    print("=" * 65)

    random.seed(42)
    octopus = GradientOctopus70()

    batch_inputs = []
    batch_targets = []
    for idx in range(20):
        x = 5.0 + idx * 1.5
        base_in = [x, x + 7.0, x - 2.0, x + 14.0, x + 2.0]
        target = [sum(base_in) / 5.0 + 10.0]
        if idx % 4 == 0:
            base_in[1] += 50.0
        batch_inputs.append(base_in)
        batch_targets.append(target)

    gaps_before = []
    for inputs, targets in zip(batch_inputs, batch_targets):
        t, t_err, gaps = octopus.forward_backward(inputs, targets)
        gaps_before.append(gaps.get("out_0", 0.0))

    print("[CPU]: Training...")
    logs = []
    epochs, nodes, gap, dt = octopus.train_batch(
        batch_inputs, batch_targets, max_iterations=30, logger_callback=logs.append)
    for l in logs:
        print(f"  {l}")
    print(f"[CPU]: Done in {dt:.2f} ms")

    gaps_after = []
    for inputs, targets in zip(batch_inputs, batch_targets):
        t, t_err, gaps = octopus.forward_backward(inputs, targets)
        gaps_after.append(gaps.get("out_0", 0.0))

    print()
    print("-" * 65)
    print(f"  {'ID':<4} | {'Noise':<10} | {'BEFORE':<12} | {'AFTER':<12}")
    print("-" * 65)
    for idx in range(20):
        noise = "+50 YES" if idx % 4 == 0 else "No"
        print(f"  {idx:<4} | {noise:<10} | {gaps_before[idx]:<12.2f} | {gaps_after[idx]:<12.4f}")

    print("-" * 65)
    avg_b = sum(gaps_before) / len(gaps_before)
    avg_a = sum(gaps_after) / len(gaps_after)
    imp = ((avg_b - avg_a) / max(avg_b, 0.001)) * 100.0
    print(f"  Avg BEFORE:  {avg_b:.2f}")
    print(f"  Avg AFTER:   {avg_a:.4f}")
    print(f"  Improvement: {imp:.1f}%")
    print(f"  Iterations:  {epochs}")
    print(f"  Nodes:       {nodes}")
    print(f"  Time:        {dt:.2f} ms")
    print("=" * 65)

if __name__ == "__main__":
    run_analytics()
