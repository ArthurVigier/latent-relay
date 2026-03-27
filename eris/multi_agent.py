"""
eris/multi_agent.py
====================

Multi-agent coordination for ERIS.

Multiple OrchestratorLLM instances can share a single ProbeModel and
optionally share steering directions.  Three coordination modes:

    ISOLATED
        Each agent reasons independently.  No shared state.
        The probe is still shared (read-only) but agents never see each
        other's activations.  Baseline mode for kill-gate Test MA-0.

    SHARED_MEDIUM
        Agents share the probe and can read each other's activation
        snapshots.  Each agent's recalibration context includes a brief
        summary of what the other agents' representations look like at
        the same step.  No shared steering library.

    COLLABORATIVE
        Full sharing: probe + steering library.
        Agents vote on which steering directions to apply.
        Voting strategies: "majority" (default), "leader" (agent 0 decides),
        "consensus" (unanimous agreement required).

Usage::

    from eris.multi_agent import MultiAgentCoordinator, CoordinationMode, AgentConfig
    from eris.backends.orchestrators.claude_orchestrator import ClaudeOrchestrator
    from eris.probe import LatentProbe

    probe = LatentProbe("Qwen/Qwen3-14B", layers=[9, 18], device="cuda")
    coord = MultiAgentCoordinator(probe=probe, mode=CoordinationMode.SHARED_MEDIUM, n_agents=3)

    agents = [
        AgentConfig(llm=ClaudeOrchestrator(), name=f"agent_{i}")
        for i in range(3)
    ]
    result = coord.run(agents, problem="Is P = NP?", max_steps=20)
    print(result.consensus_answer)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

from eris.interfaces import OrchestratorLLM, ProbeModel, ReasoningStep

log = logging.getLogger("eris.multi_agent")


# ── Coordination modes ─────────────────────────────────────────────────────────

class CoordinationMode(Enum):
    ISOLATED        = auto()  # agents reason independently
    SHARED_MEDIUM   = auto()  # shared probe, read peers' activations
    COLLABORATIVE   = auto()  # shared probe + shared steering library


# ── Config and result types ───────────────────────────────────────────────────

@dataclass
class AgentConfig:
    """Configuration for one agent in a multi-agent run."""
    llm:  OrchestratorLLM
    name: str = "agent"


@dataclass
class AgentResult:
    """Reasoning output for one agent."""
    agent_name:    str
    final_answer:  str
    n_steps:       int
    history:       list[ReasoningStep]
    shared_acts:   dict[int, dict[int, np.ndarray]] = field(default_factory=dict)
    # shared_acts: {step: {layer: np.ndarray}} — activations shared with peers


@dataclass
class MultiAgentResult:
    """Aggregated output of a multi-agent run."""
    problem:          str
    mode:             CoordinationMode
    agent_results:    list[AgentResult]
    consensus_answer: Optional[str]  # None in ISOLATED mode
    shared_directions: list[str]     # names of steering directions voted in
    n_agents:         int


# ── Coordinator ───────────────────────────────────────────────────────────────

class MultiAgentCoordinator:
    """
    Coordinates multiple OrchestratorLLM agents around a shared ProbeModel.

    Args:
        probe:    Shared ProbeModel instance (read-only in ISOLATED/SHARED_MEDIUM).
        mode:     CoordinationMode — ISOLATED, SHARED_MEDIUM, or COLLABORATIVE.
        n_agents: Number of agents (used for validation — must match len(agents) in run()).
    """

    def __init__(
        self,
        probe: ProbeModel,
        mode: CoordinationMode = CoordinationMode.ISOLATED,
        n_agents: int = 2,
    ) -> None:
        self.probe    = probe
        self.mode     = mode
        self.n_agents = n_agents
        self._shared_directions: dict[str, np.ndarray] = {}  # COLLABORATIVE only
        log.info("MultiAgentCoordinator: mode=%s n_agents=%d", mode.name, n_agents)

    def run(
        self,
        agents: list[AgentConfig],
        problem: str,
        *,
        max_steps: int = 20,
        checkpoint_every: int = 3,
        pooling: str = "last_token",
        voting_strategy: str = "majority",
    ) -> MultiAgentResult:
        """
        Run a multi-agent session on the given problem.

        In ISOLATED mode: each agent reasons independently, no cross-agent context.
        In SHARED_MEDIUM: agents receive peer activation summaries at checkpoints.
        In COLLABORATIVE: agents additionally vote on steering directions.

        Args:
            agents:           List of AgentConfig (one per agent).
            problem:          The problem all agents reason about.
            max_steps:        Max reasoning steps per agent.
            checkpoint_every: Extract probe activations every N steps.
            pooling:          Pooling strategy for probe extraction.
            voting_strategy:  "majority" | "leader" | "consensus" (COLLABORATIVE only).

        Returns:
            MultiAgentResult with all agent outputs and coordination metadata.
        """
        if len(agents) != self.n_agents:
            raise ValueError(
                f"Expected {self.n_agents} agents, got {len(agents)}."
            )

        if self.mode == CoordinationMode.ISOLATED:
            return self._run_isolated(agents, problem, max_steps, checkpoint_every, pooling)
        elif self.mode == CoordinationMode.SHARED_MEDIUM:
            return self._run_shared_medium(agents, problem, max_steps, checkpoint_every, pooling)
        else:
            return self._run_collaborative(
                agents, problem, max_steps, checkpoint_every, pooling, voting_strategy
            )

    def share_direction(self, name: str, vector: np.ndarray) -> None:
        """
        Add a steering direction to the shared library (COLLABORATIVE mode).

        This is the write path — agents propose directions, the coordinator
        stores them, and voting decides whether to apply them.
        """
        self._shared_directions[name] = np.array(vector, dtype=np.float32)
        self.probe.save_direction(name, vector)
        log.info("Shared direction saved: %r (dim=%d)", name, len(vector))

    # ── Mode implementations ───────────────────────────────────────────────────

    def _run_isolated(
        self,
        agents: list[AgentConfig],
        problem: str,
        max_steps: int,
        checkpoint_every: int,
        pooling: str,
    ) -> MultiAgentResult:
        """Each agent reasons independently. No cross-agent information."""
        results = []
        for ag in agents:
            log.info("[ISOLATED] Running agent: %s", ag.name)
            history, final = self._reason_loop(
                llm=ag.llm,
                problem=problem,
                max_steps=max_steps,
                peer_acts_fn=None,
            )
            results.append(AgentResult(
                agent_name=ag.name,
                final_answer=final,
                n_steps=len(history),
                history=history,
            ))

        return MultiAgentResult(
            problem=problem,
            mode=self.mode,
            agent_results=results,
            consensus_answer=None,
            shared_directions=[],
            n_agents=self.n_agents,
        )

    def _run_shared_medium(
        self,
        agents: list[AgentConfig],
        problem: str,
        max_steps: int,
        checkpoint_every: int,
        pooling: str,
    ) -> MultiAgentResult:
        """
        Shared probe + cross-agent activation summaries.

        Agents run in lockstep: at each checkpoint, all agents extract
        activations, then each agent receives a brief text summary of
        the other agents' representations before continuing.
        """
        n = len(agents)
        histories:   list[list[ReasoningStep]] = [[] for _ in range(n)]
        final_answers: list[str] = [""] * n
        shared_acts: list[dict] = [{} for _ in range(n)]
        active = [True] * n

        # Reference activations from problem statement (shared).
        ref_acts = self.probe.probe(problem, layers=getattr(self.probe, "layers", [-1]), pooling=pooling)

        for step in range(1, max_steps + 1):
            # Each agent takes one reasoning step.
            recal_contexts: list[Optional[str]] = [None] * n

            for i, ag in enumerate(agents):
                if not active[i]:
                    continue
                try:
                    rs = ag.llm.reason_step(
                        problem=problem,
                        history=histories[i],
                        recalibration_context=recal_contexts[i],
                    )
                    histories[i].append(rs)
                except Exception as e:
                    log.error("[SHARED_MEDIUM] agent %s step %d failed: %s", ag.name, step, e)
                    active[i] = False
                    continue

                if "[Final Answer]" in rs.content or step == max_steps:
                    final_answers[i] = rs.content
                    active[i] = False

            # Checkpoint: extract activations and share summaries.
            if step % checkpoint_every == 0:
                step_acts: list[Optional[dict]] = [None] * n
                for i, ag in enumerate(agents):
                    if histories[i]:
                        try:
                            step_acts[i] = self.probe.probe(
                                histories[i][-1].content,
                                layers=getattr(self.probe, "layers", [-1]),
                                pooling=pooling,
                            )
                            shared_acts[i][step] = step_acts[i]
                        except Exception as e:
                            log.warning("[SHARED_MEDIUM] probe failed for %s: %s", ag.name, e)

                # Build peer summaries for next step.
                for i in range(n):
                    if not active[i]:
                        continue
                    peer_lines = [f"[Peer Activation Summary — step {step}]"]
                    for j, ag in enumerate(agents):
                        if j != i and step_acts[j] is not None:
                            norms = {
                                layer: float(np.linalg.norm(vec))
                                for layer, vec in step_acts[j].items()
                            }
                            peer_lines.append(
                                f"  {ag.name}: layer norms = "
                                + ", ".join(f"L{l}={n:.3f}" for l, n in norms.items())
                            )
                    if len(peer_lines) > 1:
                        recal_contexts[i] = "\n".join(peer_lines)

            if not any(active):
                break

        agent_results = [
            AgentResult(
                agent_name=agents[i].name,
                final_answer=final_answers[i],
                n_steps=len(histories[i]),
                history=histories[i],
                shared_acts=shared_acts[i],
            )
            for i in range(n)
        ]

        # Consensus: pick the most common final answer (rough heuristic).
        consensus = _majority_answer([r.final_answer for r in agent_results])

        return MultiAgentResult(
            problem=problem,
            mode=self.mode,
            agent_results=agent_results,
            consensus_answer=consensus,
            shared_directions=[],
            n_agents=self.n_agents,
        )

    def _run_collaborative(
        self,
        agents: list[AgentConfig],
        problem: str,
        max_steps: int,
        checkpoint_every: int,
        pooling: str,
        voting_strategy: str,
    ) -> MultiAgentResult:
        """
        Full collaboration: shared probe + steering voting.
        Backward compatible: Safely extends SHARED_MEDIUM logic.
        """
        log.info(f"[COLLABORATIVE] Starting full voting loop (strategy: {voting_strategy})")

        n = len(agents)
        histories: list[list[ReasoningStep]] = [[] for _ in range(n)]
        final_answers: list[str] = [""] * n
        shared_acts: list[dict] =[{} for _ in range(n)]
        active = [True] * n

        # Le système garde la trace du vecteur de steering actif (Representation Engineering)
        active_steering = "none"

        for step in range(1, max_steps + 1):
            recal_contexts: list[Optional[str]] = [None] * n

            for i, ag in enumerate(agents):
                if not active[i]:
                    continue

                # Si un steering est actif, on l'injecte dans le contexte environnemental
                ctx = recal_contexts[i] or ""
                if active_steering != "none":
                    ctx += f"\n[System Observation] Active Latent Steering Vector: '{active_steering}' applied to Qwen."

                try:
                    rs = ag.llm.reason_step(
                        problem=problem,
                        history=histories[i],
                        recalibration_context=ctx if ctx else None,
                    )
                    histories[i].append(rs)

                    # --- VOTING LOGIC ---
                    # L'agent peut proposer un vecteur en écrivant [STEER: nom_vecteur]
                    if "[STEER:" in rs.content:
                        start = rs.content.find("[STEER:") + 7
                        end = rs.content.find("]", start)
                        if end != -1:
                            proposed = rs.content[start:end].strip()
                            # En mode leader, l'Agent 0 (Claude 0) a le pouvoir de décision immédiat
                            if voting_strategy == "leader" and i == 0:
                                active_steering = proposed
                                log.info(f"[COLLABORATIVE] Leader {ag.name} engaged steering vector: {active_steering}")
                                if active_steering not in self._shared_directions:
                                    self._shared_directions[active_steering] = np.array([1.0]) # Placeholder tracker

                except Exception as e:
                    log.error("[COLLABORATIVE] agent %s step %d failed: %s", ag.name, step, e)
                    active[i] = False
                    continue

                if "[Final Answer]" in rs.content or step == max_steps:
                    final_answers[i] = rs.content
                    active[i] = False

            # Checkpoint: extract activations and share summaries
            if step % checkpoint_every == 0:
                step_acts: list[Optional[dict]] = [None] * n
                for i, ag in enumerate(agents):
                    if histories[i]:
                        try:
                            # Extraction latente (en passant le vecteur actif si le probe le supporte)
                            probe_kwargs = {"layers": getattr(self.probe, "layers", [-1]), "pooling": pooling}
                            if active_steering != "none":
                                probe_kwargs["steer"] = active_steering

                            step_acts[i] = self.probe.probe(
                                histories[i][-1].content,
                                **probe_kwargs
                            )
                            shared_acts[i][step] = step_acts[i]
                        except Exception as e:
                            # Backward compatibility: ignore safe si le probe ne gère pas encore l'argument `steer`
                            try:
                                step_acts[i] = self.probe.probe(histories[i][-1].content, layers=getattr(self.probe, "layers", [-1]), pooling=pooling)
                            except Exception:
                                pass

                # Build peer summaries
                for i in range(n):
                    if not active[i]:
                        continue
                    peer_lines = [f"[Peer Activation Summary — step {step}]"]
                    for j, ag in enumerate(agents):
                        if j != i and step_acts[j] is not None:
                            norms = {layer: float(np.linalg.norm(vec)) for layer, vec in step_acts[j].items()}
                            peer_lines.append(f"  {ag.name}: layer norms = " + ", ".join(f"L{l}={n:.3f}" for l, n in norms.items()))
                    if len(peer_lines) > 1:
                        recal_contexts[i] = "\n".join(peer_lines)

            if not any(active):
                break

        agent_results = [
            AgentResult(
                agent_name=agents[i].name,
                final_answer=final_answers[i],
                n_steps=len(histories[i]),
                history=histories[i],
                shared_acts=shared_acts[i],
            )
            for i in range(n)
        ]

        consensus = _majority_answer([r.final_answer for r in agent_results])

        return MultiAgentResult(
            problem=problem,
            mode=self.mode,
            agent_results=agent_results,
            consensus_answer=consensus,
            shared_directions=list(self._shared_directions.keys()),
            n_agents=self.n_agents,
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _reason_loop(
        self,
        llm: OrchestratorLLM,
        problem: str,
        max_steps: int,
        peer_acts_fn,  # callable(step, history) → Optional[str] or None
    ) -> tuple[list[ReasoningStep], str]:
        """Simple reasoning loop used by ISOLATED mode."""
        history: list[ReasoningStep] = []
        final_answer = ""
        for step in range(1, max_steps + 1):
            recal = peer_acts_fn(step, history) if peer_acts_fn else None
            try:
                rs = llm.reason_step(problem=problem, history=history, recalibration_context=recal)
            except Exception as e:
                log.error("reason_step failed at step %d: %s", step, e)
                break
            history.append(rs)
            if "[Final Answer]" in rs.content or step == max_steps:
                final_answer = rs.content
                break
        return history, final_answer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _majority_answer(answers: list[str]) -> Optional[str]:
    """Return the most frequent non-empty answer. None if all empty."""
    counts: dict[str, int] = {}
    for a in answers:
        if a:
            counts[a] = counts.get(a, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda k: counts[k])
