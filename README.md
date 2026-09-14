# 🐙 Gradient Octopus v7.0 (Градиентный Осьминог)
Below is the English language.

**Gradient Octopus** — это спайковая нейронная сеть (Spiking Neural Network), которая обучается на геометрии временных задержек, а не на перемножении матриц. Сеть сама выращивает свою структуру прямо во время обучения. Алгоритм работает на любом процессоре — GPU не нужен.

---

## ⚔️ Сравнение с классическим градиентным спуском

| Критерий | Градиентный спуск (Backprop) | Осьминог v7.0 |
| :--- | :--- | :--- |
| **Что меняется** | Только веса (числа), структура заморожена | Временные задержки + рождение новых нейронов |
| **Скорость** | Тысячи эпох | 6–15 итераций |
| **Память** | Гигабайты, нужен GPU | Килобайты, хватает CPU |
| **Устойчивость к шуму** | Ломается на выбросах | Кворум отсекает шум автоматически |
| **Железо** | NVIDIA RTX | Intel i5 2014 года |

---

## 📐 Три формулы Осьминога

Ниже — математическое описание каждого шага алгоритма. Формулы записаны в формате LaTeX, GitHub их отрендерит автоматически.

---

### Формула 1 — Прямой ход с Порогом Кворума

Каждый нейрон получает входящие сигналы от своих предшественников. Время прибытия сигнала от нейрона `i` к нейрону `j` равно сумме времени активации `i` и задержки на ребре между ними:

$$ t_{\text{arrival}}(i \to j) = t(i) + \tau_{ij} $$

Нейрон `j` активируется в момент, равный среднему времени прибытия всех входящих сигналов:

$$ t(j) = \frac{1}{|I_j|} \sum_{i \in I_j} t_{\text{arrival}}(i \to j) $$

**Но с одним условием — Порог Кворума.** Если хотя бы один сигнал опаздывает относительно самого раннего больше чем на `Q = 25` тактов, он отбрасывается:

$$ I_j^{\text{valid}} = \{ i \in I_j \mid t_{\text{arrival}}(i \to j) - \min_{k} t_{\text{arrival}}(k \to j) \leq Q \} $$

Итоговая формула прямого хода:

$$ t(j) = \frac{1}{|I_j^{\text{valid}}|} \sum_{i \in I_j^{\text{valid}}} \left( t(i) + \tau_{ij} \right) $$

**Простыми словами:** «Посчитай среднее время прибытия всех сигналов, но если кто-то опоздал больше чем на 25 тактов — выкинь его из расчёта. Одна опечатка не должна ломать весь ответ».

---

### Формула 2 — Семантический зазор

После прямого хода мы знаем, во сколько сигнал **реально** пришёл в выходной нейрон: `t(j)`. После обратного хода (от цели к входам) мы знаем, во сколько он **должен был** прийти: `t_err(j)`. Разница между ними — это зазор:

$$ \Delta_j = | t(j) - t_{\text{err}}(j) | $$

Обратный ход считается аналогично прямому, но в обратном направлении — от выходов к входам, тоже с Порогом Кворума:

$$ t_{\text{err}}(j) = \frac{1}{|O_j^{\text{valid}}|} \sum_{k \in O_j^{\text{valid}}} \left( t_{\text{err}}(k) + \tau_{jk} \right) $$

**Простыми словами:** «Насколько рано или поздно ты ответил по сравнению с правильным ответом? Если зазор маленький — нейрон работает хорошо. Если большой — нужно чинить».

---

### Формула 3 — Режим Строителя (рождение нейрона)

Если зазор нейрона `j` больше порога `delta = 8.0` и Маляр (подстройка задержек) не может его убрать — Строитель рождает новый нейрон ровно посередине зазора.

Сначала находим «входной» и «выходной» концы зазора:

$$ p = \arg\min_{i \in I_j} \left( t(i) + \tau_{ij} \right) $$

$$ q = \arg\min_{k \in O_j} \left( t_{\text{err}}(k) + \tau_{jk} \right) $$

Новый нейрон `N` помещается в середину зазора:

$$ t(N) = \frac{t(j) + t_{\text{err}}(j)}{2} $$

Задержки на новых рёбрах:

$$ \tau_{pN} = \max\left(0, \; t(N) - t(p)\right) $$

$$ \tau_{Nj} = \max\left(0, \; t_{\text{err}}(j) - t(N)\right) $$

Старое ребро от `p` к `j` удаляется — сигнал теперь идёт через новую промежуточную станцию.

**Простыми словами:** «Если прямая дорога слишком длинная — поставь посередине заправку. Путь разобьётся на два коротких куска, и каждый кусок легче настроить».

---

### Бонус: Маляр (подстройка задержек)

Когда зазор меньше `delta`, Строитель спит. Вместо него работает Маляр — он аккуратно двигает задержки всех рёбер одновременно, усредняя поправку по всему батчу:

$$ \tau_{ij}^{\text{new}} = \tau_{ij}^{\text{old}} + \alpha \cdot \frac{1}{B} \sum_{b=1}^{B} \left( t_{\text{err}}^{(b)}(j) - t^{(b)}(i) - \tau_{ij}^{\text{old}} \right) $$

Где:
- `alpha = 0.3` — скорость подстройки
- `B` — размер батча (количество примеров)
- `t_err^(b)(j)` — желаемое время активации нейрона `j` на примере `b`
- `t^(b)(i)` — реальное время активации нейрона `i` на примере `b`

**Простыми словами:** «Не строй новую дорогу — просто чуть-чуть подкрути светофоры на всех перекрёстках сразу, учитывая все примеры из батча».

---

### Батч-голосование (Матрица Напряжения)

Вместо рождения нейрона после каждого примера, Осьминог собирает зазоры по всему батчу из `B` примеров и усредняет:

$$ \bar{\Delta}_j = \frac{1}{B} \sum_{b=1}^{B} \Delta_j^{(b)} $$

Нейрон рождается только для узла с максимальным средним зазором:

$$ j^* = \arg\max_j \bar{\Delta}_j $$

И только если Маляр застрял (зазор не уменьшается 3 итерации подряд).

**Простыми словами:** «Не лечи каждого пациента отдельно — собери статистику по всей больнице, найди самую большую проблему, и лечи только её одну».

---

### Прунинг (удаление слабых узлов)

Каждый раз при рождении нового нейрона старый узел ослабляется:

$$ w_j = w_j \cdot \beta $$

Где `beta = 0.5`. Если вес падает ниже `w_min = 0.05` — узел и все его рёбра удаляются из графа полностью.

**Простыми словами:** «Если дорогой никто не пользуется — закрой её и разбери рельсы».

---

## 🧪 Результаты краш-теста

Тест: 20 примеров, 5 с шумом (+50 тактов на входе), цель = среднее входов + 10.

| Параметр | Значение |
| :--- | :--- |
| Средний зазор ДО обучения | 1.15 тактов |
| Средний зазор ПОСЛЕ обучения | 0.36 тактов |
| Улучшение | **68.8%** |
| Итераций | **6** |
| Нейронов | **6** (ни одного нового — Маляр справился сам) |
| Время | **7.77 мс** (Intel i5-4690, 2014 год) |

Шумные примеры получили зазор **0.12** — кворум отсёк выброс, сеть даже не заметила шум.

---

## 🚀 Как запустить

Нужен Python 3.x. Зависимостей нет — только стандартная библиотека.

**Графический интерфейс:**

```bash
python octopus_v7.py



# 🐙 Gradient Octopus v7.0 (Gradient Octopus)

**Gradient Octopus** is a spiking neural network (Spiking Neural Network) that learns based on the geometry of time delays, rather than matrix multiplication. The network grows its own structure directly during training. The algorithm works on any processor — no GPU is needed.

---

## ⚔️ Comparison with classic gradient descent

| Criterion | Gradient Descent (Backprop) | Octopus v7.0 |
| :--- | :--- | :--- |
| **What changes** | Only the weights (numbers), the structure is frozen | Time delays + the birth of new neurons |
| **Speed** | Thousands of epochs | 6–15 iterations |
| **Memory** | Gigabytes, GPU required | Kilobytes, CPU is sufficient |
| **Noise resistance** | Breaks on outliers | Quorum automatically filters out noise |
| **Hardware** | NVIDIA RTX | Intel i5 2014 |

---

## 📐 Three Octopus formulas

Below is the mathematical description of each step of the algorithm. The formulas are written in LaTeX format; GitHub will render them automatically.

---

### Formula 1 — Direct Path with Quorum Threshold

Each neuron receives incoming signals from its predecessors. The arrival time of a signal from neuron `i` to neuron `j` is equal to the sum of the activation time of `i` and the delay on the edge between them:

$$ t_{\text{arrival}}(i \to j) = t(i) + \tau_{ij} $$

Neuron `j` activates at the moment equal to the average arrival time of all incoming signals:

$$ t(j) = \frac{1}{|I_j|} \sum_{i \in I_j} t_{\text{arrival}}(i \to j) $$

**But with one condition — the Quorum Threshold.** If at least one signal is delayed relative to the earliest one by more than `Q = 25` clock cycles, it is discarded:

$$ I_j^{\text{valid}} = \{ i \in I_j \mid t_{\text{arrival}}(i \to j) - \min_{k} t_{\text{arrival}}(k \to j) \leq Q \} $$

The final formula for the direct path:

$$ t(j) = \frac{1}{|I_j^{\text{valid}}|} \sum_{i \in I_j^{\text{valid}}} \left( t(i) + \tau_{ij} \right) $$

**In simple terms:** “Calculate the average arrival time of all signals, but if someone is late by more than 25 clock cycles, exclude them from the calculation. One typo shouldn’t ruin the entire answer.”

---

### Formula 2 — Semantic Gap

After the forward pass, we know when the signal **actually** arrived at the output neuron: `t(j)`. After the backward pass (from the target to the inputs), we know when it **should have** arrived: `t_err(j)`. The difference between them is the gap:

$$ \Delta_j = | t(j) - t_{\text{err}}(j) | $$

The reverse move is calculated similarly to the direct move, but in the opposite direction — from outputs to inputs, also with the Quorum Threshold:

$$ t_{\text{err}}(j) = \frac{1}{|O_j^{\text{valid}}|} \sum_{k \in O_j^{\text{valid}}} \left( t_{\text{err}}(k) + \tau_{jk} \right) $$

**In simple words:** “How early or late did you respond compared to the correct answer?” If the gap is small, the neuron works well. If it's big, you need to fix it."

---

### Formula 3 — Builder Mode (birth of a neuron)

If the gap of the neuron `j` is greater than the threshold `delta = 8.0' and the Painter (delay adjustment) cannot remove it, the Builder generates a new neuron exactly in the middle of the gap.

"First" we find the "input" and "output" ends of the gap.:

$$ p = \arg\min_{i \in I_j} \left( t(i) + \tau_{ij} \right) $$

$$ q = \arg\min_{k \in O_j} \left( t_{\text{err}}(k) + \tau_{jk} \right) $$

The new neuron `N` is placed in the middle of the gap:

$$ t(N) = \frac{t(j) + t_{\text{err}}(j)}{2} $$

Delays on the new edges:

$$ \tau_{pN} = \max\left(0, \; t(N) - t(p)\right) $$

$$ \tau_{Nj} = \max\left(0, \; t_{\text{err}}(j) - t(N)\right) $$

The old edge from `p` to `j` is removed — the signal now goes through the new intermediate station.

**In simple terms:** “If the straight road is too long, put a gas station in the middle. The path will break into two short sections, and each section will be easier to adjust.”

---

### Bonus: Painter (delay adjustment)

When the gap is smaller than `delta`, the Builder is asleep. Instead, the Painter works — he carefully adjusts the delays of all edges simultaneously, averaging the correction across the entire batch:

$$ \tau_{ij}^{\text{new}} = \tau_{ij}^{\text{old}} + \alpha \cdot \frac{1}{B} \sum_{b=1}^{B} \left( t_{\text{err}}^{(b)}(j) - t^{(b)}(i) - \tau_{ij}^{\text{old}} \right) $$

Where:
- `alpha = 0.3` — the adjustment rate
- `B` — batch size (number of examples)
- `t_err^(b)(j)` — the desired activation time of neuron `j` for example `b`
- `t^(b)(i)` — the actual activation time of neuron `i` for example `b`

**In simple terms:** “Don’t build a new road — just tweak the traffic lights at all the intersections at once, taking into account all the examples from the batch.”

---

### Batch voting (Voltage Matrix)

Instead of the neuron being born after each example, Octopus collects the gaps across the entire batch of `B` examples and averages them:

$$ \bar{\Delta}_j = \frac{1}{B} \sum_{b=1}^{B} \Delta_j^{(b)} $$

The neuron is born only for the node with the maximum average gap:

$$ j^* = \arg\max_j \bar{\Delta}_j $$

And only if the Painter is stuck (the gap does not decrease for 3 consecutive iterations).

**In simple words:** “Don’t treat each patient separately — collect statistics for the entire hospital, find the biggest problem, and treat only that one.”

---

### Pruning (removing weak nodes)

Each time a new neuron is born, the old node is weakened:

$$ w_j = w_j \cdot \beta $$

Where `beta = 0.5`. If the weight drops below `w_min = 0.05`, the node and all its edges are completely removed from the graph.

**In simple terms:** “If no one uses the road, close it and dismantle the rails.”

---

## 🧪 Crash test results

Test: 20 examples, 5 with noise (+50 input cycles), goal = average inputs + 10.

| Parameter | Value |
| :--- | :--- |
| Average gap BEFORE training | 1.15 cycles |
| Average gap AFTER training | 0.36 cycles |
| Improvement | **68.8%** |
| Iterations | **6** |
| Neurons | **6** (no new ones — the painter handled it himself) |
| Time | **7.77 ms** (Intel i5-4690, 2014) |

Noisy examples received a gap of **0.12** — the quorum cut off the outlier, and the network didn’t even notice the noise.

---

## 🚀 How to run

Python 3.x is required. There are no dependencies — only the standard library.

**Graphical interface:**

```bash
python octopus_v7.py
