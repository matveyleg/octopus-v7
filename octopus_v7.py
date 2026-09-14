# -*- coding: utf-8 -*-
import tkinter as tk
import random
import time

ALPHABET = "abvgdeezhijklmnoprstufhccssqyeua "

def text_to_spikes(text, max_inputs=5):
    words = text.lower().split()
    spikes = []
    for word in words[:max_inputs]:
        if len(word) > 0:
            idx = ALPHABET.find(word[0])
            if idx == -1: idx = 15
            spikes.append(float(idx + 1.0))
    while len(spikes) < max_inputs:
        spikes.append(10.0)
    return spikes


class GradientOctopus70:
    def __init__(self, num_inputs=5, num_outputs=1):
        self.epsilon = 0.5
        self.delta = 8.0
        self.alpha = 0.3
        self.beta = 0.5
        self.w_min = 0.05
        self.N_max = 80
        self.quorum_threshold = 25.0

        self.inputs = [f"in_{i}" for i in range(num_inputs)]
        self.outputs = [f"out_{k}" for k in range(num_outputs)]
        self.hidden = []
        self.edges = {}
        self.weights = {}

        for out_node in self.outputs:
            self.edges[out_node] = {}
            for in_node in self.inputs:
                self.edges[out_node][in_node] = random.uniform(5.0, 15.0)

        for node in self.inputs + self.outputs:
            self.weights[node] = 1.0
        self.tentacle_counter = 0

    def get_all_nodes(self):
        return self.inputs + self.hidden + self.outputs

    def get_in_edges(self, j):
        return self.edges.get(j, {})

    def get_out_edges(self, j):
        out_nodes = {}
        for target, inputs in self.edges.items():
            if j in inputs:
                out_nodes[target] = inputs[j]
        return out_nodes

    def forward_backward(self, input_times, target_times):
        all_nodes = self.get_all_nodes()
        t = {node: float("inf") for node in all_nodes}
        for i, in_node in enumerate(self.inputs):
            t[in_node] = input_times[i]

        changed = True
        while changed:
            changed = False
            for j in all_nodes:
                if j in self.inputs:
                    continue
                in_links = self.get_in_edges(j)
                if not in_links:
                    continue
                raw_times = [t[i] + tau for i, tau in in_links.items() if t[i] != float("inf")]
                if raw_times:
                    base_min = min(raw_times)
                    valid_times = [tx for tx in raw_times if tx - base_min <= self.quorum_threshold]
                    if valid_times:
                        avg_time = sum(valid_times) / len(valid_times)
                        if avg_time < t[j]:
                            t[j] = avg_time
                            changed = True

        t_err = {node: float("inf") for node in all_nodes}
        for k, out_node in enumerate(self.outputs):
            t_err[out_node] = target_times[k]

        changed = True
        while changed:
            changed = False
            for j in all_nodes:
                if j in self.outputs:
                    continue
                out_links = self.get_out_edges(j)
                if not out_links:
                    continue
                raw_times = [t_err[k] + tau for k, tau in out_links.items() if t_err[k] != float("inf")]
                if raw_times:
                    base_min = min(raw_times)
                    valid_times = [tx for tx in raw_times if tx - base_min <= self.quorum_threshold]
                    if valid_times:
                        avg_time = sum(valid_times) / len(valid_times)
                        if avg_time < t_err[j]:
                            t_err[j] = avg_time
                            changed = True

        gaps = {}
        for j in all_nodes:
            if t[j] == float("inf") or t_err[j] == float("inf"):
                gaps[j] = 0.0
            else:
                gaps[j] = abs(t[j] - t_err[j])
        return t, t_err, gaps

    def birth_neuron(self, worst_node, t, t_err):
        in_links = self.get_in_edges(worst_node)
        if not in_links:
            return None
        p = min(in_links.keys(), key=lambda i: t[i] + in_links[i])
        self.tentacle_counter += 1
        N_new = f"H_Node_{self.tentacle_counter}"
        self.hidden.append(N_new)
        self.weights[N_new] = 1.0
        self.edges[N_new] = {}
        mid_time = (t[worst_node] + t_err[worst_node]) / 2.0
        self.edges[N_new][p] = max(0.0, mid_time - t[p])
        self.edges[worst_node].pop(p, None)
        self.edges[worst_node][N_new] = max(0.0, t_err[worst_node] - mid_time)
        self.weights[worst_node] *= self.beta
        return N_new

    def train_batch(self, batch_inputs, batch_targets, max_iterations=30, logger_callback=None):
        start_time = time.time()
        final_epoch = 0
        final_gap = 0.0
        stagnation_count = 0
        prev_gap = float("inf")

        for iteration in range(max_iterations):
            final_epoch = iteration
            tension_matrix = []
            all_t = []
            all_t_err = []

            for inputs, targets in zip(batch_inputs, batch_targets):
                t, t_err, gaps = self.forward_backward(inputs, targets)
                tension_matrix.append(gaps)
                all_t.append(t)
                all_t_err.append(t_err)

            avg_gaps = {}
            all_nodes = self.get_all_nodes()
            for node in all_nodes:
                node_gaps = [g.get(node, 0.0) for g in tension_matrix]
                avg_gaps[node] = sum(node_gaps) / len(node_gaps)

            output_gaps = [avg_gaps.get(o, 0.0) for o in self.outputs]
            final_gap = max(output_gaps) if output_gaps else 0.0

            if logger_callback:
                logger_callback(f"Iter {iteration:02d} | Out gap: {final_gap:.2f} | Nodes: {len(all_nodes)}")

            if final_gap < self.epsilon:
                break

            # PAINTER: batch-averaged delay adjustment
            edge_deltas = {}
            for inputs, targets in zip(batch_inputs, batch_targets):
                t, t_err, gaps = self.forward_backward(inputs, targets)
                for j in all_nodes:
                    if j in self.inputs:
                        continue
                    if gaps[j] > self.epsilon:
                        in_links = self.get_in_edges(j)
                        if in_links:
                            for i, tau in in_links.items():
                                if t[i] != float("inf") and t_err[j] != float("inf"):
                                    desired_tau = t_err[j] - t[i]
                                    delta_tau = self.alpha * (desired_tau - tau)
                                    key = (j, i)
                                    if key not in edge_deltas:
                                        edge_deltas[key] = []
                                    edge_deltas[key].append(delta_tau)

            for (j, i), deltas in edge_deltas.items():
                avg_delta = sum(deltas) / len(deltas)
                self.edges[j][i] = max(0.0, self.edges[j][i] + avg_delta)

            # BUILDER: structural change only if Painter is stuck
            if abs(prev_gap - final_gap) < 0.5:
                stagnation_count += 1
            else:
                stagnation_count = 0

            if stagnation_count >= 3 and final_gap > self.delta:
                candidate_nodes = [n for n in all_nodes if n not in self.inputs]
                worst_node = max(candidate_nodes, key=lambda n: avg_gaps.get(n, 0.0))
                worst_ex_idx = 0
                worst_val = -1
                for idx, gaps in enumerate(tension_matrix):
                    g = gaps.get(worst_node, 0.0)
                    if g > worst_val:
                        worst_val = g
                        worst_ex_idx = idx
                new_node = self.birth_neuron(worst_node, all_t[worst_ex_idx], all_t_err[worst_ex_idx])
                if logger_callback and new_node:
                    logger_callback(f"   Builder: {new_node} on {worst_node}")
                stagnation_count = 0

            prev_gap = final_gap

            # PRUNING
            for h_node in list(self.hidden):
                if self.weights.get(h_node, 1.0) < self.w_min:
                    self.hidden.remove(h_node)
                    if h_node in self.edges:
                        del self.edges[h_node]
                    for target in list(self.edges.keys()):
                        if h_node in self.edges[target]:
                            del self.edges[target][h_node]
                    if logger_callback:
                        logger_callback(f"   Pruned: {h_node}")

        elapsed = (time.time() - start_time) * 1000.0
        return final_epoch, len(self.get_all_nodes()), final_gap, elapsed


class OctopusV7GUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("MAI: Gradient Octopus Engine v7.0")
        self.window.geometry("580x560")
        self.window.configure(bg="#0b0c10")

        self.octopus = GradientOctopus70()
        tk.Label(self.window, text="OCTOPUS BATCH-EVOLUTION CORE v7.0",
                 fg="#45f3ff", bg="#0b0c10", font=("Arial", 12, "bold")).pack(pady=15)

        self.btn_run = tk.Button(self.window, text="RUN BATCH TRAINING + CRASH TEST",
                                 font=("Arial", 10, "bold"), bg="#1f2833", fg="#00f5d4",
                                 width=42, height=2, command=self.run_process)
        self.btn_run.pack(pady=10)

        tk.Label(self.window, text="Iteration log:", fg="#8d99ae", bg="#0b0c10",
                 font=("Arial", 8)).pack(anchor="w", padx=30)
        self.log_text = tk.Text(self.window, fg="#00f5d4", bg="#13131a",
                                width=64, height=12, font=("Courier", 9))
        self.log_text.pack(pady=5)

        self.stat_frame = tk.LabelFrame(self.window, text=" FINAL STATISTICS ",
                                        fg="#ffb703", bg="#0b0c10",
                                        font=("Arial", 9, "bold"), padx=10, pady=5)
        self.stat_frame.pack(fill="x", padx=20, pady=10)
        self.stat_label = tk.Label(self.stat_frame, text="Press START to run analysis",
                                   fg="#8d99ae", bg="#0b0c10",
                                   font=("Courier", 9), justify="left")
        self.stat_label.pack(anchor="w")
        self.window.mainloop()

    def log_msg(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.window.update()

    def run_process(self):
        self.log_text.delete("1.0", tk.END)
        self.log_msg("[SYSTEM]: Building test batch...")
        random.seed(42)
        self.octopus = GradientOctopus70()

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

        self.log_msg("[SYSTEM]: 20 examples, 5 with noise (+50 ticks)")
        self.log_msg("-" * 55)

        epochs, nodes, gap, dt = self.octopus.train_batch(
            batch_inputs, batch_targets, logger_callback=self.log_msg)
        self.log_msg("-" * 55)
        self.log_msg("[DONE]")

        stat = (f"  Iterations:  {epochs}\n"
                f"  Nodes:       {nodes}\n"
                f"  Final gap:   {gap:.4f} ticks\n"
                f"  Time:        {dt:.2f} ms")
        self.stat_label.config(text=stat, fg="#00f5d4")


if __name__ == "__main__":
    OctopusV7GUI()
