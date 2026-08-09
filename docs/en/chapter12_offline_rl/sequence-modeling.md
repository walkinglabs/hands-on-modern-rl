# 10.2 Decision Transformer, Trajectory Transformer, and Diffuser

> [12.1](./intro) covered offline RL's three conservative approaches within the Bellman framework — BCQ, CQL, and IQL. This section takes a different route: **abandoning Bellman entirely** and reformulating RL as **conditional sequence generation**. Decision Transformer models trajectories directly with GPT, Trajectory Transformer uses beam search, and Diffuser uses a diffusion model — together, the three point toward the paradigm shift of "RL as sequence modeling."

## Decision Transformer and RL as Sequence Modeling

The previous three sections all worked within the Bellman framework — constraining actions, constraining Q, adding a BC regularizer. **Decision Transformer (Chen et al. 2021) abandons Bellman entirely**, reformulating RL as a **conditional sequence generation** problem and modeling trajectories directly with GPT.

### Return-to-Go: Treating Return as a Condition

DT's core insight: under a supervised learning framework, in a trajectory $\tau = (s_1, a_1, r_1, s_2, a_2, r_2, \ldots, s_T, a_T, r_T)$, every action $a_t$ has a **natural target** — the cumulative return from time $t$ onward:

$$\hat{R}_t = \sum_{t'=t}^{T} r_{t'}$$

This is called the **return-to-go**. Given $\hat{R}_t$ and $s_t$, predicting $a_t$ becomes ordinary conditional supervised learning.

DT reorganizes the trajectory into a sequence of triples:

$$\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \ldots, \hat{R}_T, s_T, a_T$$

Each timestep contributes three tokens: (RTG, state, action). A GPT-style causal transformer then models the sequence autoregressively:

$$\pi_\theta(a_t \mid \hat{R}_t, s_t, a_{t-1}, \ldots) = \text{Transformer}(\hat{R}_{1:t}, s_{1:t}, a_{1:t-1})$$

```python
class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_dim, n_heads, n_layers,
                 max_ep_len=4096):
        super().__init__()
        # Three embedding layers: RTG, state, and action each map to hidden_dim
        self.embed_rtg  = nn.Linear(1, hidden_dim)
        self.embed_state = nn.Linear(state_dim, hidden_dim)
        self.embed_action = nn.Linear(act_dim, hidden_dim)
        self.embed_ln = nn.LayerNorm(hidden_dim)
        # Positional encoding: timestep embedding
        self.pos_emb = nn.Embedding(max_ep_len, hidden_dim)
        # GPT backbone
        self.transformer = GPT(
            d_model=hidden_dim, n_heads=n_heads, n_layers=n_layers,
            # Key: each timestep occupies 3 tokens, so the attention mask must match
            attn_pdrop=0.1, resid_pdrop=0.1
        )
        # Action prediction head (regression, assuming continuous actions)
        self.action_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, rtg, states, actions, timesteps):
        B, T, _ = states.shape
        # Embed and interleave: (R1, s1, a1, R2, s2, a2, ...)
        rtg_emb   = self.embed_rtg(rtg)
        state_emb = self.embed_state(states) + self.pos_emb(timesteps)
        action_emb = self.embed_action(actions)

        # Stack the three into (B, 3T, H), in order RTG, state, action
        stacked = torch.stack([rtg_emb, state_emb, action_emb], dim=1)
        stacked = stacked.permute(0, 2, 1, 3).reshape(B, 3 * T, -1)
        stacked = self.embed_ln(stacked)

        # Causal attention: each token can only see the past
        h = self.transformer(stacked)
        # Take the output at state positions to predict the corresponding action
        h_states = h[:, 1::3, :]  # indices 1, 4, 7, ...
        return self.action_head(h_states)  # Regress to continuous actions

    @torch.no_grad()
    def act(self, state, target_rtg, history, t):
        # At inference time, treat the target RTG as a "prompt" and generate actions autoregressively
        rtg_seq = torch.cat([history.rtg, target_rtg[None]], dim=0)[-self.K:]
        s_seq   = torch.cat([history.states, state[None]], dim=0)[-self.K:]
        a_seq   = history.actions[-self.K - 1:-1]  # offset by one
        t_seq   = torch.arange(len(s_seq))
        pred_a = self.forward(rtg_seq, s_seq, a_seq, t_seq)
        return pred_a[-1]  # Take the prediction for the last timestep
```

### Pure Supervision

DT's training loss is simply the MSE of continuous action regression (or cross-entropy for discrete actions):

$$\mathcal{L} = \mathbb{E}_{\tau \sim \mathcal{D}}\left[\sum_t \|\hat{a}_t - a_t\|^2\right]$$

**No Bellman, no Q-learning, no temporal difference.** The entire training process is identical to training a GPT: scan trajectories, predict the next token. This property lets DT plug directly into the LLM training stack — data loading, AdamW, cosine schedules, gradient checkpointing all carry over unchanged.

### Using RTG as a Control Variable

At deployment, DT needs no argmax over Q. You simply **specify a target RTG** (say, the expert-level score for the environment), and DT autoregressively generates actions that drive the cumulative return toward that target:

```python
target_return = 9000  # HalfCheetah expert-level
state = env.reset()
history = TrajectoryBuffer()
for t in range(max_steps):
    action = model.act(state, target_return, history, t)
    next_state, reward, done, _ = env.step(action)
    history.append(state, action, reward)
    state = next_state
    # Key: subtract the actual reward from RTG each step, so it tracks the "remaining target still to reach"
    target_return -= reward
```

This mechanism is elegant: RTG works as a **control variable** — turning it up or down produces policies at different performance levels. Empirically, DT **matches or exceeds** CQL/IQL on Atari, MuJoCo, and Key-to-Door.

### Why Does DT Work?

This is one of the most contested questions in the offline RL community. From the traditional RL perspective, **without Bellman backups it should be impossible to learn an optimal policy for long-term return** — because the supervisory signal can only come from the trajectories that were actually collected. DT's answer: **when the dataset is rich enough, the trajectories themselves already carry the optimality information.**

- The dataset contains expert trajectories (high RTG), medium trajectories (moderate RTG), and random trajectories (low RTG)
- Given a high target RTG, the conditional distribution $p(a \mid \hat{R}_{\text{high}}, s)$ that the transformer learns naturally favors high-return actions
- This amounts to a form of **retrieval-based policy learning** — at its core, it imitates "trajectories that once achieved a similar RTG"

Formally, the policy DT learns can be written as:

$$\pi_\theta(a \mid s, \hat{R}) \propto \exp\left(-\frac{1}{2\sigma^2}\|a - f_\theta(s, \hat{R})\|^2\right)$$

where $f_\theta$ is the transformer's regression output. As $\sigma \to 0$, this degenerates to the deterministic policy $a = f_\theta(s, \hat{R})$. Notice the relationship between this distribution and $\pi_\beta$:

$$\pi_\theta(a \mid s, \hat{R}) \approx \pi_\beta(a \mid s, \text{return} \approx \hat{R})$$

In other words, what DT learns is the behavior policy's conditional distribution given a specified return. This is exactly why DT cannot surpass the best policy present in the dataset — it never combines the good parts of two suboptimal trajectories.

This observation spawned a wave of follow-up work: RL via supervised learning in the online RL setting, in-context RL (Algorithm Distillation), Star-Vector, Eyre et al.'s "language modeling is all you need for RL," and more.

::: details DT's Limitations

1. **Can only learn the best policy present in the data** — if the dataset contains no expert trajectories, no target RTG, however high, can produce expert behavior
2. **Weak stitching ability** — traditional offline RL can "stitch" the good parts of two suboptimal trajectories together into a better policy (subtrajectory stitching); because DT is pure supervision, it cannot achieve this kind of compositional generalization
3. **Sensitive to RTG choice** — setting the target RTG too high produces incoherent actions, setting it too low produces overly conservative behavior
   :::

## Trajectory Transformer and Diffuser

After DT, the "RL as sequence modeling" line of work branched out quickly. Two representative examples: Trajectory Transformer models the entire trajectory as a token sequence and does inference with beam search; Diffuser uses a diffusion model to generate the complete trajectory directly.

### Trajectory Transformer: Discretization + Beam Search

Janner et al. 2021 discretize RTG, state, action, and reward all into tokens, then train a standard transformer to predict the next token:

$$p_\theta(\tau) = \prod_{t=1}^{T} p_\theta(s_t, a_t, r_t \mid s_{<t}, a_{<t}, r_{<t})$$

At inference time, beam search maximizes trajectory probability (optionally with a reward constraint added). TT's characteristics:

- Discretizing continuous quantities avoids the regression problem, but the token count explodes (every dimension of the state must be discretized)
- Beam search inference is slow (it must expand multiple candidate trajectories)
- Advantage: it can do **planning** — future reward constraints can be injected explicitly during search, which amounts to implicit model-based RL

### Diffuser: Generating Trajectories with a Diffusion Model

Janner et al. 2022 brought diffusion models into RL. A trajectory $\tau \in \mathbb{R}^{T \times (d_s + d_a)}$ is treated as a high-dimensional, image-like object, and a diffusion model is trained on it:

$$\min_\theta \; \mathbb{E}_{\tau, t, \epsilon}\left[\|\epsilon - \epsilon_\theta(\tau_t, t)\|^2\right]$$

where $\tau_t$ is the trajectory after noise has been added at diffusion timestep $t$, and $\epsilon_\theta$ is the denoising network (typically a 1D temporal U-Net or a transformer). At inference time, denoising proceeds step by step starting from pure noise, producing the complete trajectory.

Diffuser's killer feature is **classifier-free guidance**: during training, the condition (state, reward function) is randomly dropped, so the model learns both the conditional and unconditional distributions simultaneously:

$$\tilde{\epsilon}_\theta = (1 + w) \cdot \epsilon_\theta(\tau_t, t, c) - w \cdot \epsilon_\theta(\tau_t, t)$$

where $c$ is the condition (e.g., "maximize future reward") and $w$ controls the strength of conditioning. This lets Diffuser **guide trajectory generation with a reward function** — at its core, it rewrites "maximizing a value function" as "sampling from a probability model."

### Comparing DT / TT / Diffuser

| Dimension                                 | Decision Transformer      | Trajectory Transformer                       | Diffuser                                  |
| ----------------------------------------- | ------------------------- | -------------------------------------------- | ----------------------------------------- |
| Modeling target                           | Policy conditioned on RTG | Joint distribution over the whole trajectory | Diffusion model over the whole trajectory |
| Discretization                            | No                        | Yes (every state dimension discretized)      | No                                        |
| Inference method                          | Autoregressive sampling   | Beam search                                  | Iterative denoising                       |
| Planning ability                          | Weak (implicit)           | Strong (explicit)                            | Strong (conditional generation)           |
| Stitching ability                         | Weak                      | Medium                                       | Strong                                    |
| Inference speed                           | Fast                      | Slow                                         | Medium (needs dozens of denoising steps)  |
| Compatibility with the LLM training stack | Strong (closest to GPT)   | Strong                                       | Weak (different architecture)             |

## Section Summary

Decision Transformer casts RL as conditional sequence generation: given a return-to-go, generate actions autoregressively. This paradigm shift merges the RL training stack with the LLM training stack. Trajectory Transformer goes further and brings in planning through beam search; Diffuser generates complete trajectories with a diffusion model.

The next section, [12.3 Offline RL Experiments and the LLM Perspective](./experiments), pulls the lens back to the LLM era — you'll find that DPO is, at its core, a special case of offline RL.
