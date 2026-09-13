# Thinkbook H1. Cumulative Quantization Error in Autoregressive KV Cache

Status. Draft, unvalidated.
Subject. PM-KVQ (arXiv 2505.18610, ICLR 2026), Section 3.1 and the claim of "large cumulative error".
Reference implementation. `pm_kvq/quantization/quantizer/quantizer.py`, `pm_kvq/quantization/methods/pm_kvq/progressive/progressive_quantized_cache.py`.
Scope. This document derives a model of how KV cache quantization error evolves with decode position, states the falsifiable predictions that follow, and defines the measurement that discriminates between them. It does not evaluate PM-KVQ as a method and does not reproduce the paper's benchmarks.

---

## 1. The claim under examination

The paper asserts that quantizing the KV cache at every decoding step produces error that accumulates, and that accumulated error is the mechanism behind long-CoT degradation. The assertion is stated qualitatively. This thinkbook converts it into a growth law with a measurable exponent.

The question in one line. Given a per-step quantization noise floor that is stationary in $t$, does the observed deviation $\varepsilon(T)$ at decode position $T$ grow, saturate, or decay.

Three outcomes are possible a priori, and the naive reading of "accumulation" assumes the first without argument.

---

## 2. Notation and the quantizer

Let $d$ be head dimension, $H$ the number of KV heads, $L$ the number of transformer blocks, $g$ the quantization group size (128 in the reference implementation), $b$ the bit-width.

**Definition 1 (asymmetric uniform quantizer).**
For a group $\mathbf{x} \in \mathbb{R}^{g}$ with $x_{\max} = \max_j x_j$ and $x_{\min} = \min_j x_j$, define range $R = x_{\max} - x_{\min}$, scale and zero point

$$
S \;=\; \frac{R}{2^{b}-1}, \qquad Z \;=\; \left\lceil -\frac{x_{\min}}{S} \right\rfloor .
$$

The quantize and dequantize composition is

$$
Q_b(x) \;=\; S\Big(\operatorname{clamp}\big(\lceil x/S \rfloor + Z,\; q_{\min},\, q_{\max}\big) - Z\Big),
\qquad q_{\min}=0,\; q_{\max}=2^{b}-1 .
$$

This matches `UntrainableQuantizer.fake_quant` with `round_zeros=True`, `symmetric=False`.

**Definition 2 (per-element error).** Write $\delta = Q_b(x) - x$.

**Lemma 1 (noise floor).** Away from clipping, $|\delta| \le S/2$. Under the high-resolution model $\delta \sim \mathcal{U}(-S/2,\,S/2)$, independent across elements,

$$
\mathbb{E}[\delta] = 0, \qquad
\sigma^2 \;\equiv\; \operatorname{Var}(\delta) \;=\; \frac{S^{2}}{12} \;=\; \frac{R^{2}}{12\,(2^{b}-1)^{2}} .
$$

**Corollary 1 (bit-width scaling).** For $b \ge 4$, $(2^b-1)^2 \approx 4^{b}$, so

$$
\sigma^{2}(b) \;\approx\; \frac{R^{2}}{12}\,4^{-b},
\qquad
10\log_{10}\frac{\sigma^{2}(b)}{\sigma^{2}(b+1)} \;=\; 10\log_{10} 4 \;\approx\; 6.02 \text{ dB per bit.}
$$

Numerically, moving from 16-bit to 2-bit multiplies the noise power by

$$
\left(\frac{2^{16}-1}{2^{2}-1}\right)^{2} = \left(\frac{65535}{3}\right)^{2} \approx 4.77\times 10^{8} \;\;(\approx 86.8 \text{ dB}).
$$

This number sets the stakes. Any temporal effect must be weighed against a static factor of $10^{8}$ in injected noise power.

---

## 3. Single-step injection

At decode step $t$ the cache stores perturbed tensors

$$
\hat{\mathbf{k}}_t = \mathbf{k}_t + \boldsymbol{\delta}^{K}_t,
\qquad
\hat{\mathbf{v}}_t = \mathbf{v}_t + \boldsymbol{\delta}^{V}_t,
$$

with $\mathbb{E}\|\boldsymbol{\delta}^{K}_t\|^2 = d\,\sigma_K^2$ and $\mathbb{E}\|\boldsymbol{\delta}^{V}_t\|^2 = d\,\sigma_V^2$.

The reference implementation preserves the first 128 and the most recent 128 tokens in INT16, so the injection applies on the index set $\mathcal{Q} = \{128 < t \le T-128\}$. This is a boundary condition on the sums below, not a change to the mechanism.

---

## 4. Propagation through the attention readout

Let $\mathbf{q}_T$ be the query at step $T$, and define logits and attention weights

$$
z_{T,t} = \frac{\mathbf{q}_T^{\top}\mathbf{k}_t}{\sqrt{d}},
\qquad
a_{T,t} = \frac{e^{z_{T,t}}}{\sum_{s \le T} e^{z_{T,s}}},
\qquad
\mathbf{o}_T = \sum_{t \le T} a_{T,t}\,\mathbf{v}_t .
$$

Perturbed logits carry $\Delta z_{T,t} = \mathbf{q}_T^{\top}\boldsymbol{\delta}^{K}_t/\sqrt{d}$.

**Lemma 2 (softmax first order).** The softmax Jacobian is $\partial a_t / \partial z_s = a_t(\mathbb{1}[t=s] - a_s)$, giving

$$
\Delta a_{T,t} \;\approx\; a_{T,t}\Big(\Delta z_{T,t} - \textstyle\sum_{s} a_{T,s}\Delta z_{T,s}\Big)
\;=\; a_{T,t}\big(\Delta z_{T,t} - \overline{\Delta z}_T\big).
$$

**Proposition 1 (output perturbation, two paths).**

$$
\Delta\mathbf{o}_T \;\approx\;
\underbrace{\sum_{t} a_{T,t}\,\boldsymbol{\delta}^{V}_t}_{\text{value path}}
\;+\;
\underbrace{\sum_{t} a_{T,t}\big(\Delta z_{T,t} - \overline{\Delta z}_T\big)\,\mathbf{v}_t}_{\text{key path}} .
$$

The key path is mean-centred by construction, which is the formal statement of why the Key cache is the more sensitive tensor under a fixed $\sigma$. The centring removes the common-mode component, leaving the variance of the logit perturbation to act directly on the attention distribution.

---

## 5. The averaging term and why "accumulation" is not automatic

**Definition 3 (effective attention support).**

$$
T_{\mathrm{eff}}(T) \;=\; \frac{1}{\sum_{t} a_{T,t}^{2}} \;=\; \frac{1}{\|\mathbf{a}_T\|_2^{2}}, \qquad 1 \le T_{\mathrm{eff}} \le T .
$$

**Lemma 3 (correlated sum).** Let the per-step errors share a correlation coefficient $\rho \in [0,1]$ across positions, elementwise. Then

$$
\mathbb{E}\Big\|\sum_{t} a_{T,t}\boldsymbol{\delta}_t\Big\|^{2}
= d\sigma^{2}\Big[(1-\rho)\sum_t a_{T,t}^{2} + \rho\Big(\sum_t a_{T,t}\Big)^{2}\Big]
= d\sigma^{2}\Big[\frac{1-\rho}{T_{\mathrm{eff}}(T)} + \rho\Big],
$$

using $\sum_t a_{T,t} = 1$.

**Corollary 2 (three regimes of the readout).**

$$
\rho = 0 \;\Rightarrow\; \mathbb{E}\|\Delta\mathbf{o}_T\|^2 = \frac{d\sigma^{2}}{T_{\mathrm{eff}}(T)} \;\;\text{(decaying in }T\text{ when attention spreads)},
$$
$$
\rho = 1 \;\Rightarrow\; \mathbb{E}\|\Delta\mathbf{o}_T\|^2 = d\sigma^{2} \;\;\text{(flat in }T\text{)}.
$$

Neither regime grows. A readout that averages independent noise over a widening support suppresses that noise. Growth requires a mechanism outside the readout.

This is the first non-obvious consequence of the derivation, and it is what makes H1 worth testing rather than assuming.

---

## 6. The feedback recursion

The mechanism that remains is the autoregressive loop. The hidden state at step $t$ is computed from $\mathbf{o}_t$, and $\mathbf{k}_{t+1}, \mathbf{v}_{t+1}$ are computed from that hidden state. Error written into the cache at step $t$ re-enters as signal at step $t+1$.

Let $\varepsilon_T = \mathbb{E}\|\Delta\mathbf{o}_T\|^{2}$ and let $\gamma$ denote the block-to-block gain of the residual stream with respect to state perturbation, so that an inherited deviation of energy $\varepsilon$ arrives at the next readout with energy $\gamma^{2}\varepsilon$.

**Model 1 (injection plus inheritance).**

$$
\varepsilon_T \;=\; \underbrace{c_T}_{\text{fresh injection}} \;+\; \gamma^{2}\sum_{t<T} a_{T,t}\,\varepsilon_t,
\qquad c_T = d\,\sigma^{2}(T)\Big[\tfrac{1-\rho}{T_{\mathrm{eff}}} + \rho\Big].
$$

The inherited term carries weight $a_{T,t}$ rather than $a_{T,t}^{2}$ because propagated deviations are deterministic functions of past deviations and do not cancel as independent draws do.

**Theorem 1 (closed form under uniform mixing).** Approximate $a_{T,t}\approx 1/T$ and let $S(T)=\sum_{t\le T}\varepsilon_t$. Model 1 becomes the linear ODE

$$
\frac{dS}{dT} \;=\; c(T) + \frac{\gamma^{2}}{T}S(T).
$$

With integrating factor $T^{-\gamma^{2}}$,

$$
\frac{d}{dT}\Big(T^{-\gamma^{2}}S\Big) = c(T)\,T^{-\gamma^{2}}
\;\;\Longrightarrow\;\;
S(T) = T^{\gamma^{2}}\Big[S_0 + \int_{T_0}^{T} c(u)\,u^{-\gamma^{2}}\,du\Big].
$$

For constant $c$ and $\gamma^{2}\neq 1$,

$$
S(T) = \frac{c}{1-\gamma^{2}}\,T + A\,T^{\gamma^{2}},
\qquad
\boxed{\;\varepsilon(T) = \frac{c}{1-\gamma^{2}} + A\gamma^{2}\,T^{\,\alpha}, \quad \alpha \equiv \gamma^{2}-1 \;}
$$

For $\gamma^{2}=1$ the integral gives $\int_{T_0}^{T} c\,u^{-1}du = c\ln(T/T_0)$, hence

$$
S(T) = cT\ln(T/T_0) + AT, \qquad \varepsilon(T) = c\ln(T/T_0) + c + A .
$$

**Corollary 3 (the three regimes, restated as a single exponent).**

$$
\gamma < 1 \;\Rightarrow\; \alpha < 0 \;\Rightarrow\; \varepsilon(T)\to \frac{c}{1-\gamma^{2}} \;\text{(saturation)},
$$
$$
\gamma = 1 \;\Rightarrow\; \alpha = 0 \;\Rightarrow\; \varepsilon(T) \sim c\ln T \;\text{(marginal, logarithmic)},
$$
$$
\gamma > 1 \;\Rightarrow\; \alpha > 0 \;\Rightarrow\; \varepsilon(T) \sim T^{\alpha} \;\text{(power law growth)}.
$$

**Consequence for measurement.** A log-log plot of $\varepsilon$ against $T$ is linear with slope $\alpha$, and the gain is recovered as

$$
\gamma = \sqrt{\alpha + 1}.
$$

The paper's qualitative claim is the statement $\alpha > 0$. That statement is now falsifiable with a single regression.

---

## 7. Temporal sensitivity, and why early tokens dominate

**Theorem 2 (Green's function of Model 1).** An impulse injection of energy $c_0$ at position $t_0$ contributes to $S$ the homogeneous mode $c_0\,t_0^{-\gamma^{2}}T^{\gamma^{2}}$, hence contributes to the observable

$$
G(T, t_0) \;=\; \frac{\partial}{\partial T}\Big[c_0 t_0^{-\gamma^{2}}T^{\gamma^{2}}\Big]
\;=\; \frac{c_0\,\gamma^{2}}{t_0}\left(\frac{T}{t_0}\right)^{\alpha}.
$$

**Corollary 4.** $G$ is monotonically decreasing in $t_0$ for every $\alpha > -1$, that is for every $\gamma^2 > 0$. Noise injected early is amplified more than noise injected late, through the $1/t_0$ prefactor alone, and additionally through $(T/t_0)^{\alpha}$ when $\alpha>0$.

This is the structural argument for progressive quantization. It holds independently of the sign of $\alpha$, which makes it the more robust of the two claims in Section 3.1 of the paper.

**Theorem 3 (optimal bit schedule).** Let the memory budget be $\int_{T_0}^{T} b(t)\,dt \le B$ and let $c(t) = \kappa\,4^{-b(t)}$ from Corollary 1. Minimising $\varepsilon(T) \propto T^{\alpha}\int c(u)u^{-\gamma^{2}}du$ under the budget gives the Lagrangian

$$
\mathcal{L} = \kappa\int 4^{-b(t)}t^{-\gamma^{2}}dt + \lambda\Big(\int b(t)\,dt - B\Big),
$$
$$
\frac{\partial \mathcal{L}}{\partial b(t)} = -\kappa\ln 4\cdot 4^{-b(t)}t^{-\gamma^{2}} + \lambda = 0
\;\;\Longrightarrow\;\;
4^{-b(t)} = \frac{\lambda\,t^{\gamma^{2}}}{\kappa\ln 4},
$$
$$
\boxed{\;b^{*}(t) \;=\; \log_4\!\frac{\kappa\ln 4}{\lambda} \;-\; \gamma^{2}\log_4 t\;}
$$

The optimal bit-width falls linearly in $\log t$, with slope $-\gamma^{2}$ bits per factor of four in position.

**Open check.** The reference schedule shrinks $16 \to 8 \to 4 \to 2$ at position doublings, which is geometric in $b$ rather than linear in $\log t$. The two shapes coincide only over a narrow range. Whether the difference is material is a measurement, recorded here as unresolved.

---

## 8. Bit-width shrinking is error-neutral

The paper's Theorem D.1 establishes that

$$
X_b = \Big\lfloor (2^{2b}-2^{b}+1)\big(X_{2b}+2^{b-1}\big) \Big\rfloor \gg 3b
$$

is exactly equal to dequantizing the $2b$-bit tensor and requantizing to $b$ bits, given $Z_b = Z_{2b}$ and $S_b = (2^{b}+1)S_{2b}$.

**Consequence for this model.** Shrinking introduces no error beyond the $b$-bit floor $\sigma^{2}(b)$. In the language of Model 1, progressive quantization changes $c(t)$ from a constant $\kappa 4^{-b_{\text{final}}}$ into a staircase that begins far below it, without adding a transition penalty. The entire benefit is the Green's function weighting of Theorem 2, and nothing else.

The ablation in Table 4 of the paper is consistent with this. Direct Right Shift and Modified Right Shift are not floor-preserving and lose 32.09 and 15.42 points of pass@1 respectively.

---

## 9. Falsifiable predictions

**P1.** With the autoregressive feedback loop broken, $\varepsilon(T)$ does not grow. Formally $\alpha_{\text{tf}} \le 0$ within fit error.

**P2.** With the loop intact, $\varepsilon(T)$ grows as a power law with $\alpha_{\text{fr}} > 0$, and the log-log fit has $R^{2} \ge 0.9$.

**P3.** The gap $\alpha_{\text{fr}} - \alpha_{\text{tf}}$ is attributable to feedback gain and yields $\gamma = \sqrt{\alpha_{\text{fr}}+1} > 1$.

**P4.** Under $\rho \approx 0$ and broken feedback, the measured $\varepsilon(T)$ tracks $d\sigma^{2}/T_{\mathrm{eff}}(T)$, with $T_{\mathrm{eff}}$ computed directly from the attention maps as $1/\|\mathbf{a}_T\|_2^2$.

**P5.** Injecting noise of fixed total energy at early positions produces larger terminal deviation than injecting the same energy at late positions, with ratio predicted by $G(T,t_0^{\text{early}})/G(T,t_0^{\text{late}})$ from Theorem 2.

**P6.** Holding the schedule fixed and sweeping $b$, terminal deviation follows $6.02$ dB per bit as in Corollary 1, with departures indicating clipping rather than high-resolution behaviour.

---

## 10. Experimental protocol

Model. A small Qwen-family checkpoint, consistent with prior dissections in `qwen3-qk-norm` and `qwen3-gqa-mechanism`. Full precision reference in BF16.

Quantizer. Reimplemented from Definition 1 rather than imported, so the noise floor is known analytically rather than inherited. Group size 128, asymmetric, per-channel for Key and per-token for Value.

Observable. Normalised deviation at each decode position,

$$
\varepsilon(T) \;=\; \frac{\mathbb{E}\big\|\mathbf{o}^{\text{quant}}_T - \mathbf{o}^{\text{fp}}_T\big\|_2^{2}}{\mathbb{E}\big\|\mathbf{o}^{\text{fp}}_T\big\|_2^{2}} .
$$

Normalisation removes scale drift in the residual stream, which would otherwise be confounded with growth.

**Arm A, teacher forced.** Both the full precision model and the quantized model consume an identical fixed token sequence. The generated token never depends on the perturbed output. Feedback is off. Isolates the readout term of Proposition 1.

**Arm B, free running.** The quantized model generates its own continuation. Feedback is on. Alignment against the reference is by position index, with divergence in token identity recorded separately as a secondary observable.

The contrast between Arm A and Arm B is the experiment. Everything else is held fixed.

**Arm C, impulse.** Quantize only a window of positions $[t_0, t_0+w]$ and hold all other positions in BF16. Sweep $t_0$. Measures $G(T,t_0)$ directly and tests P5 without relying on the uniform-mixing approximation of Theorem 1.

**Instrumented quantities per run.** Per-position $\varepsilon(T)$, per-position $T_{\mathrm{eff}}(T)$ from attention maps, per-group realised $\sigma^{2}$ against the predicted $R^{2}/12(2^{b}-1)^{2}$, clipping rate as fraction of elements hitting $q_{\min}$ or $q_{\max}$.

**Positions.** Log-spaced sampling of $T$ over at least two decades, since the prediction is a power law and linear sampling wastes resolution at small $T$.

**Repetitions.** $N \ge 16$ sequences per configuration, with $\varepsilon(T)$ reported as mean and interquartile range across sequences.

---

## 11. Pass criteria and falsifiers

Recorded before execution.

| Prediction | Pass criterion | Falsifier |
|---|---|---|
| P1 | $\alpha_{\text{tf}} \le 0.05$ | $\alpha_{\text{tf}} \ge 0.2$ with $R^2 \ge 0.9$ |
| P2 | $\alpha_{\text{fr}} \ge 0.2$, $R^{2}\ge 0.9$ | $\alpha_{\text{fr}} \le 0.05$ |
| P3 | $\alpha_{\text{fr}} - \alpha_{\text{tf}} \ge 0.15$ | gap within noise of zero |
| P4 | Pearson $r \ge 0.8$ between $\varepsilon$ and $1/T_{\mathrm{eff}}$ in Arm A | $r \le 0.3$ |
| P5 | $G$ ratio within $2\times$ of prediction | monotonicity in $t_0$ reversed |
| P6 | slope within $[5.5, 6.5]$ dB per bit for $b\ge 4$ | slope outside $[4,8]$ |

**If P2 fails.** The paper's cumulative error framing does not hold at this model scale, and the observed long-CoT degradation is attributable to the static noise floor of Corollary 1 rather than to accumulation. Progressive quantization would then be justified by memory utilisation alone, and the Green's function argument of Theorem 2 would reduce to the $1/t_0$ prefactor.

**If P1 fails.** The uniform mixing approximation or the independence assumption of Lemma 3 is wrong, and Lemma 3 needs replacement before Theorem 1 can be trusted.

**Cost of skipping.** Without $\alpha$ and $\gamma$ measured, any bit-width chosen for a KV cache payload is chosen without a growth model, and terminal fidelity at a target horizon cannot be predicted before the run.

---

## 12. Bridge to the ESCP per-byte cost term

This section is the reason the thinkbook exists in this form. It is stated as a derivation, not as a result.

Let a transferred KV slice contain $n$ elements quantized to $b$ bits. Payload size is $n b / 8$ bytes. Under the three-term energy model,

$$
E(b) \;=\; E_{\text{wake}} \;+\; N_{\text{frame}}(b)\,E_{\text{frame}} \;+\; \frac{n b}{8}\,E_{\text{byte}},
\qquad
N_{\text{frame}}(b) = \left\lceil \frac{n b/8}{P_{\max}} \right\rceil,
$$

with $P_{\max}$ the payload ceiling of the carrier.

Distortion of the transferred representation follows Corollary 1,

$$
D(b) \;=\; \frac{R^{2}}{12}\,4^{-b}.
$$

Treating $N_{\text{frame}}$ as locally constant between ceiling steps and forming $J = E(b) + \lambda D(b)$,

$$
\frac{dJ}{db} = \frac{n}{8}E_{\text{byte}} - \lambda\,\frac{R^{2}\ln 4}{12}\,4^{-b} = 0
\;\;\Longrightarrow\;\;
\boxed{\;b^{*} = \log_4\!\left(\frac{2\lambda R^{2}\ln 4}{3\,n\,E_{\text{byte}}}\right)}
$$

**Reading.** Energy is linear in $b$. Distortion is exponential in $b$. The stationary point is therefore unique and the knee is well defined. The bit-width that minimises joules at a fixed fidelity target is a function of $E_{\text{byte}}$, which is exactly the term the harness measures.

**Two open couplings.**

First, $N_{\text{frame}}(b)$ is a step function, so the true optimum is the better of $b^{*}$ and the largest $b$ that does not cross a fragmentation boundary. The comparison requires $E_{\text{frame}}$, which the frame overhead probe supplies.

Second, if $\alpha > 0$ is confirmed and the receiving node continues generation from the transferred cache, then $D$ is not the terminal quantity. The terminal quantity is $\varepsilon(T)$ under Theorem 1, and $b^{*}$ shifts upward by an amount that depends on the continuation horizon. For single-shot transfer with no continuation, $D(b)$ suffices and $\alpha$ is irrelevant. Which case applies is a design decision, not a measurement, and it is unresolved.

---

## 13. What this thinkbook does not claim

It does not claim that $\alpha > 0$ holds. That is P2, and P2 is untested.

It does not claim that small-model behaviour transfers to the 7B to 70B range the paper evaluates. The recursion of Model 1 contains $\gamma$, which is a property of the residual stream and plausibly depends on depth.

It does not establish that the uniform mixing approximation $a_{T,t}\approx 1/T$ is adequate. Arm C exists because that approximation is the weakest step in Theorem 1.

It does not treat the RoPE calibration argument of Section 3.3 of the paper. That is a separate hypothesis with a separate mechanism.

---

## 14. Artifacts

| Artifact | Content |
|---|---|
| `results/h1_arm_a.csv` | position, $\varepsilon$, $T_{\mathrm{eff}}$, clipping rate, sequence id |
| `results/h1_arm_b.csv` | as above, free running |
| `results/h1_arm_c.csv` | $t_0$, window width, terminal $\varepsilon$ |
| `results/h1_fit.json` | $\alpha_{\text{tf}}$, $\alpha_{\text{fr}}$, $R^{2}$, recovered $\gamma$, confidence intervals |
| `figures/h1_loglog.png` | $\varepsilon$ against $T$, both arms, fitted lines |
| `figures/h1_green.png` | measured $G(T,t_0)$ against Theorem 2 |

## 15. References

Liu et al. PM-KVQ, Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs. ICLR 2026. arXiv 2505.18610.
Liu et al. KIVI, tuning-free asymmetric 2bit quantization for KV cache. arXiv 2402.02750.
Chen et al. Extending context window of large language models via positional interpolation. arXiv 2306.15595.
Su et al. RoFormer, enhanced transformer with rotary position embedding. Neurocomputing 568, 2024.
