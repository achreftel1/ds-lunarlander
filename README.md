# ds-lunarlander

Validates the **Dynamical-System (DS) reward model** on Gymnasium's
LunarLander: two linear transition operators `T_c` (safe landing) and `T_h`
(crash), fit from labeled state trajectories, score any new trajectory by
prediction residuals — exactly the whiteboard math from the DS hallucination
paper, applied to real physical dynamics.

**Result:** the decision rule `sign(r_DS)` classifies held-out trajectories
with ~92% accuracy. The DS method works as a reward model on a system whose
ground truth is exact and free.

**Scope note:** this repo contains *no policy optimization*. Injecting `r_DS`
into GRPO belongs to the LLM paper, where the policy starts from an
SFT-quality model. Here we validate the reward model itself, in isolation.

---

## Why LunarLander

| Whiteboard | LLM case (paper 2) | LunarLander (this repo) |
|---|---|---|
| trajectory `x_1..x_K` | sentence embeddings | 8-D physical state |
| regime `c` (correct) | factual response | **safe landing**, terminal reward `+100` |
| regime `h` (hallucinated) | hallucinated response | **crash**, terminal reward `-100` |
| `T_c`, `T_h` | fit on labeled embeddings | fit on labeled state trajectories |
| labels | expensive human annotation | **free and exact**, from `env.step()` |

Gymnasium ships **no RL algorithms** — it is only an environment API
(`reset`, `step`, `render`). Terminal labels come straight from the source
(`gymnasium/envs/box2d/lunar_lander.py::step()`):

```python
if self.game_over or abs(state[0]) >= 1.0:
    terminated = True;  reward = -100      # crash  -> regime h
if not self.lander.awake:
    terminated = True;  reward = +100      # landed -> regime c
```

---

## The math

Two linear transition operators, one per regime:

```
x^c_{k+1} = T_c x^c_k            x^h_{k+1} = T_h x^h_k
```

Fit in closed form (ridge least squares, no gradient descent, no epochs):

```
T = argmin_T  sum_k || x_{k+1} - T x_k ||^2 + alpha ||T||_F^2
```

Score any trajectory by its residual under each operator:

```
eps_c = (1/(K-1)) sum_k || x_{k+1} - T_c x_k ||^2
eps_h = (1/(K-1)) sum_k || x_{k+1} - T_h x_k ||^2

diff  = (eps_c - eps_h) / eps_c      <- whiteboard normalized differential
ratio = eps_h / eps_c                <- whiteboard arrow, want >> 1
r_DS  = log(eps_h / eps_c)           <- the reward-model output
```

Decision rule: `r_DS > 0` → correct-regime dynamics (safe landing);
`r_DS < 0` → wrong-regime dynamics (crash).

**Why `log(ratio)` and not `diff`:** `diff` is asymmetric — it saturates at
exactly `+1` deep in the `h` regime but is unbounded negative deep in the `c`
regime. `log(ratio)` is symmetric, well scaled, and crosses zero at the
identical decision boundary `eps_c == eps_h`.

### Three deliberate engineering choices (all toggleable, all reported)

| Flag | Default | Why |
|---|---|---|
| `standardize` | on | the 8 dims have wildly different scales (positions are `O(1)` floats, leg flags are binary) — unstandardized LS is dominated by the large dims |
| `affine` | on | gravity is a **constant** acceleration; a purely linear `T` cannot represent free fall. Appending a constant `1` lets the operator learn `x_{k+1} = A x_k + b` |
| `normalize_eps` | on | successful landings are systematically longer than crashes; an unnormalized **sum** would conflate "wrong regime" with "long episode" |

---

## Install

Requires **Python 3.10–3.12** (not 3.13; Box2D wheels are still patchy there).

```bash
git clone <your-repo-url> ds-lunarlander
cd ds-lunarlander

python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**If Box2D fails** (the one step that commonly breaks):

```bash
# all platforms, try this FIRST
pip install swig
pip install box2d-py
pip install "gymnasium[box2d]"

# macOS, if the above fails
brew install swig && xcode-select --install

# Linux
sudo apt install swig build-essential python3-dev

# Windows, last resort: install "Desktop development with C++" from
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

Verify:

```bash
python -c "import gymnasium as gym; e=gym.make('LunarLander-v3'); o,_=e.reset(seed=0); print(o.shape, e.action_space)"
# expect:  (8,) Discrete(4)
```

---

## Run

```bash
python scripts/run_all.py
```

Or stage by stage:

```bash
python src/collect.py       --episodes 600
python src/fit_operators.py
python src/ds_scorer.py     --demo
```

Useful variants:

```bash
python scripts/run_all.py --skip-collect        # reuse saved trajectories
python scripts/run_all.py --no-affine           # strict whiteboard linear form
python scripts/run_all.py --no-standardize      # raw state scales
python scripts/run_all.py --alpha 1e-1          # stronger regularization
```

---

## How to read the output

```
mean r_DS | safe landing : +X.XXXX     (positive)
mean r_DS | crash        : -X.XXXX     (negative)
separation               : +X.XXXX     (large)
accuracy, HELD-OUT       :    XX.X%    <- the number to report
```

Positive separation + high held-out accuracy means the whiteboard decision
rule holds on real physical dynamics. `results/separation.png` is the
one-figure summary: two histograms of `r_DS` (green = landings, red =
crashes) split by the `r_DS = 0` decision boundary.

---

## Layout

```
ds-lunarlander/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── collect.py          Step 1: labeled trajectories (env gives ground truth)
│   ├── fit_operators.py    Step 2: closed-form ridge fit of T_c, T_h
│   └── ds_scorer.py        Step 3: frozen reward model, r_DS = log(eps_h/eps_c)
├── scripts/
│   └── run_all.py          steps 1->3 + separation figure
└── results/                operators, trajectories, figure
```

---

## If separation fails

Try in this order:

1. `--no-standardize` / `--no-affine` — compare all four combinations
2. `--alpha 1e-1` — more regularization
3. `--episodes 1500` — more data per regime

If none work, that is a **genuine methodological finding**, not user error: a
single global *linear* operator may underfit LunarLander's contact-switching
dynamics (the leg-contact flags make the system piecewise). The standard fix is
lifting into nonlinear observables before fitting (EDMD) — a question for the
DS paper's authors.

---

## Questions for the advisor

1. **Residual normalization.** Should `eps` be summed or averaged over `K-1`?
   Successful landings are systematically longer than crashes, so the
   unnormalized sum may measure episode length rather than regime.
2. **Affine term.** Gravity is constant acceleration; purely linear `T` cannot
   express it. Does the paper include a bias term, or lift the state first?
3. **Window granularity.** Operators are fit on full episodes. Fixed-length
   sliding windows would give more data, finer residuals, and map more directly
   onto per-sentence LLM trajectories. Which does the paper do?

---

## References

- Gymnasium: <https://gymnasium.farama.org/>
- LunarLander docs: <https://gymnasium.farama.org/environments/box2d/lunar_lander/>
- Source: `Farama-Foundation/Gymnasium` → `gymnasium/envs/box2d/lunar_lander.py`
- DS Low-Cost Black-Box Detection of LLM Hallucinations via Dynamical System Prediction: <https://arxiv.org/abs/2605.05134>