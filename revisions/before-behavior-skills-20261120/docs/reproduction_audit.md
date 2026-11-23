# Reproduction audit (2026-11-20, Asia/Seoul)

## Source provenance and existing work

The starting directories `/home/jeonges/acd-experiments` and `/home/jeonges/proposed`
were not Git working trees. `git status --short` returned "not a git repository".
The latter is the user's existing copy of the former; the research extension is
added there. Existing source hashes are preserved in `preexisting_files_sha256.json`.
No pre-existing ACD implementation or result is replaced. The original source
checkouts remain independently versioned and their SHA values are recorded in
`../research_sources.lock.json`:

| Source | Commit |
|---|---|
| cage-challenge-4 | `8c3c50ca54b176c2de199847944e8dcc035497e3` |
| llms-are-acd | `81ff29a0d8f7a7184cafecc62c39945d03acdda6` |
| cage-4-submission | `2afd652d80ce9d4051a07c23c2538f3dec6bb6c6` |
| Skill-Pro | `3be7a9be2d4c024d132efe394d537404eba7e4c8` |

Paper versions: [ACD v2](https://arxiv.org/abs/2505.04843v2),
[Skill-Pro v3](https://arxiv.org/abs/2602.01869v3).

## ACD actual call paths

Paths below are relative to `third_party/llms-are-acd/` unless marked otherwise.

| Responsibility | Actual code |
|---|---|
| LLM adapter | `CybORG/Agents/LLMAgents/llm_agent.py:DefenderAgent.get_action` → `LLMDefenderPolicy.compute_single_action` |
| Observation | `llm_adapter/obs_formatter.py:format_observation`, `_format_suspicious_activity`, `_format_comm_vector_message` |
| Prompt | `config/prompts/acd2025/base.yml`; `llm_policy.py:_init_strategy_prompt`; current-observation user message, no previous conversation |
| Response | `llm_policy.py:extract_action`; substring parsing, unrecognized response → Sleep; no Monitor branch |
| Model | `llm_adapter/model_manager.py:BackendFactory.create_backend`; eagerly imports OpenAI/HF/Weave even for dummy |
| KEEP | `CybermonicAgents/cage4.py:load`, `InductiveGraphPPOAgent.get_action`; same five weight SHA256s as the Cybermonic repo |
| Red | `SimpleAgents/FiniteStateRedAgent.py`, `AggressiveFSMAgent.py`, `StealthyFSMAgent.py`, `ImpactFSMAgent.py`, `DegradeServiceFSMAgent.py` |
| Communication | `LLMAgents/comm_vector/CommVectorGenerator.py:create_comm_message`; first 5 owner bits, 2 severity bits, 1 busy bit |
| Combined execution | `Evaluation/Cybermonics/submission.py:CombinedWrapper.step/reset` calls both PhaseWrapper and GraphWrapper on one simulator; mixed mode can double-step/reset and overwrite observations |
| Official evaluation | `third_party/cage-challenge-4/CybORG/Evaluation/evaluation.py:run_evaluation` |

The official evaluator constructs one environment then calls `wrapped.reset()` for
each episode without reseeding. It computes `mean(rew.values())`, adds that once
per nonterminal turn, and uses `statistics.stdev` over episode totals. Its terminal
check occurs BEFORE appending the terminal reward. The scenario's `determine_done`
returns true at `step_count >= steps-1`. Thus a configured length of 500 can mean
499 executed joint transitions and 498 scored transitions with this wrapper.
This is explicitly represented by the `official` scoring profile; a corrected
terminal-inclusive score is an optional COMMON comparison change, never silently
substituted for the official metric. Raw terminal rewards remain in the trace.

The paper reports 2 episodes of nominal length 500. It does not publish all seeds
or establish the precise across-Red weighting/sample-vs-population convention for
its headline -493/95.9 and -2547.2/498.8 figures. Those values and 45.2/4704.6s are
stored only as `paper_reference`, not as a Qwen measurement. Per-episode code uses
sample standard deviation. Paper timings and cross-Red summary uncertainty are
kept separate from the new explicitly defined wall-clock interval.

Original prompts omit Sleep from the AVAILABLE ACTIONS list although parsing accepts
it. Monitor is an automatic end-of-turn action in SimulationController, and is not
an LLM response choice in the original parser. `paper-compatible` preserves this;
`controlled-comparison` adds the SAME explicit Sleep rule to E0–E3, preserving
automatic Monitor. Qwen non-thinking chat templating is shared across all arms.

## Skill-Pro actual call graph and discrepancies

Paths relative to `third_party/Skill-Pro/`:

| File / callable | Observed implementation |
|---|---|
| `main.py:parse_arguments/main` | constructs `run.SkillMDP`; MDP/SMDP and `wo_sg`, `wo_ppo`, `wo_score` switches |
| `run.py:SkillMDP.run_skill_mdp` | environment rollouts → EMA task baseline (alpha .1) → maturity → evolution → maintenance |
| `run.py:run_single_episode` | TextArena/ALFWorld-specific calls; per-agent skill persists, `_should_terminate` LLM meta-controller, maximum age 8; a five-iteration warmup branch exists but is inactive by default (`self.warm_up_flag = False`, line 44) |
| `data_structures.py:Skill/Experience` | I/policy/termination, frequency/gain/version; Experience has structured transitions AND a separate trajectory string |
| `pool_managers.py:ExperiencePool/GoldenExperiencePool` | FIFO and top-return heap; instantiated in run.py but NOT used to drive the current evolution path |
| `Skills/skill_pool.py:select_skill/_select_by_llm` | fixed epsilon exploration including NONE, LLM XML choice, gain/maturity admission; no learned neural selector |
| `Skills/skill_evolution.py:_generate_semantic_gradient` | initiation/execution/termination diagnosis from `experience.trajectory` |
| `...:evolve/_run_semantic_update` | per-experience gradients → batch aggregation + refinement/new-skill JSON; versioned skills |
| `...:run_skill_evolution_with_verification` | threshold 6, best-of-3, old/new logprob, clipped score, accept J > margin, append candidate |
| `...:_compute_skill_logprobs` | vLLM policy or HF loss route; configured evolver can differ from acting model |
| `Skills/loss.py:calculate_log_prob_batch` | comment says sum, actual returns MEAN target-token logprob; separate prompt/target truncation to half context |
| `utils/local_llm.py:compute_logprob_batch` | vLLM SUM, text concatenation boundary assumption, top-1 fallback to minimum available logprob; overflow can return 0; HF import `.loss` refers to absent `utils/loss.py` |
| `Skills/skill_pool.py:maintain` | warmup/maturity and LCB-based capacity management, semantic dedup with MiniLM, NOT the paper's simple non-positive online-score pruning |

Important paper/code differences:

1. The paper uses return-to-go minus a running baseline and an expectation of each
   trajectory's mean PPO surrogate. The current code instead uses a uniform
   `(trajectory_return - EMA_baseline)/trajectory_length` advantage and a mean over
   all transitions, clamping log ratios to [-10,10]. Both are named distinctly in
   this extension (`official_code` versus `paper_rtg`); neither is passed off as
   the other. E2/E3 always use the SAME selected mode.
2. Experience.trajectory in run.py contains the FINAL observation/result summary,
   while its `transitions` contains actions but no step reward/next observation.
   The CAGE4 extension feeds actual time-ordered before/after/action/status/skill
   records to semantic-gradient chunking; all chunks are included, no silent
   context truncation or final-return-only causal diagnosis.
3. Original evolution places the same multi-skill full episode in every skill's
   buffer, then scores every transition for every parent. CAGE4 experiences are
   separated by Blue AND skill version/active span; mixed-agent evidence is not
   silently assigned to another skill.
4. Original `wo_ppo` still computes PPO scores and selects the best by them before
   unconditional acceptance. The explicit CAGE4 no-gate ablation does NOT call
   logprob at all. It admits structurally valid candidates in generation order.
5. Original loss paths disagree (sum/mean), can approximate missing tokens, and do
   not use Qwen chat templates identically for policy and counterfactual scoring.
   The new backend teacher-forces the exact archived completion token IDs,
   including EOS when emitted, with the SAME frozen Qwen weights/tokenizer/chat
   template/temperature. Full-support sampling uses top_p=1/top_k=0; no truncated
   top-k fallback. These are response-sequence log probabilities, including the
   emitted reason, NOT a claim to marginalize all reason strings into P(action).
6. Selection's algorithm remains fixed; this is not a parametric PPO/GRPO/LoRA
   learner. No Qwen weight has requires_grad=True or optimizer steps.

## Reuse / adaptation / new contribution

- Unmodified reuse: frozen source checkout classes `Skill`, `Experience`, FIFO and
  Golden pools, LLM skill selector, score statistics, priority/evaluation sampling,
  semantic gradients, batch aggregation, no-gradient summarization, and reference
  maintenance methods. The encoder dependency is injected through a local MiniLM
  service; the original source files are not edited.
- CAGE4 adaptation: source-preserving environment bridge, joint action execution,
  per-Blue state, full trajectories, strict local policy client, exact token-ID
  scoring, resumable per-episode evaluator, metric/report paths.
- Algorithm completion/corrections: explicit official-code versus paper RTG gate,
  fail-closed probability errors, filtered skill/agent trajectories and genuine
  off-switch ablations. Each mode is recorded in configs and results.
- Proposed E3 contribution: observable-only phase/message/status/repetition and
  executable action/target verification. E2 contains no E3 verification. This
  extension changes external skill selection/validity, never CAGE4 rewards.

Actual test counts, executed episodes, measured performance and incomplete work
are written separately in the final execution report, not inferred from features.

## Additional compatibility and numerical checks

The unchanged public KEEP checkpoints expect 192 features, including three
legacy communication features per peer. The paper's 8-bit wire format is not
directly compatible. `ResearchEnv._message_features` explicitly projects severity
>=2 to compromised, >=1 to scanned, and any message bit to received. Native own
features and the weights remain fixed. All E0–E3 hybrid arms use this SAME
projection. Zero-valued messages cannot distinguish no threat from no message.
The separate `keep_native` profile retains the source message interpretation.
Neither profile is labeled an unchanged reproduction with unpublished retrained
paper-compatible KEEP weights.

Full-forward BF16 scoring initially disagreed with sampled generation by about
0.116 log-probability units. It was replaced before the 500-step learning runs
by incremental cache-aligned teacher forcing. Two actual policy generations then
agreed to 0 and 3.5762786865234375e-7 (tolerance 1e-4). The validation artifact is
`../research_outputs/scoring-validation.json`; approximated scores are not used
as successful Gate verification.

`paper_algorithm.py` separately implements the active-step gain / invocation-count
score and pruning path. A never-invoked candidate is provisionally retained with
zero measured gain; this explicit cold-start rule is a completion choice, not a
claim about an unspecified paper detail. The paper RTG baseline is an EMA over
complete, unique joint time series, not duplicated Blue/Skill observations.
The default experiment remains `official_code` with the actual source LCB manager.

During integration the source `update_stats` input denominator was corrected to
each Blue/episode's skill invocation count instead of the whole batch's count.
The two long trainers were resumed from preserved first-episode checkpoints before
Evolution, so the correction applies to their learned pools. Incomplete attempts
are archived separately. New invocations record implementation hashes; the
earliest process manifests predate that logging addition.

The actual unmodified evaluator was invoked for two no-blue episodes. Its own
mean/sample deviation exactly matched the bridge ledger, with 499 executed and
498 scored transitions each. This validates reward arithmetic and reset-stream
behavior; it is not a Qwen performance measurement.

## Final executed checks

- 42 regression/research tests passed; 20 preserved ACD/config/test files have no hash changes.
- 49 completed experimental rollouts, 6 distinct iteration-validation episodes,
  and 2 actual original-evaluator episodes were recorded (57 CAGE4 episodes total).
  Short integration rollouts are counted separately from nominal-500 evaluations.
- Five runtime core files (reward machine/calculator, simulator controller,
  scenario generator, official evaluator) are byte-identical to official CAGE4.
- All 24 locked policy/encoder files still match their download SHA256 values.
- Gate/generation parity covered 32 unique real decisions: 16 archived generation
  scores and 16 older decisions reproduced with identical completion tokens.
  Maximum absolute sequence-logprob difference was 7.152557373046875e-7.
- E1 interrupted/replayed prefixes matched for 480 joint transitions. After E2's
  candidates were rejected, E1/E2 matched on all 998 joint evaluation transitions
  (4,990 Blue records), including responses, actions, messages and rewards.
- The 12 main/transfer evaluation episodes shared one policy model revision,
  tokenizer-template hash, precision and decoder configuration. Training and
  frozen evaluation Pool snapshots matched; the transfer Pool was unchanged.
- GPU 1–3 inference servers were stopped after the experiments. The latest default
  server remains on GPU 0 / localhost:8018 with zero trainable policy parameters.

Actual measured rewards, failed improvement hypotheses and unexecuted final-study
cells are reported in `execution_report.md`. The code now permits an explicit
model-ID/revision change for a separate future experiment, rejects unexpected or
trainable backend identities, and never substitutes a model on failure. Every
experiment executed for this report used the pinned Qwen3-8B weights.
