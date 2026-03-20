"""Falsification layer: pre-generation path filtering.

Critical design: falsify paths BEFORE hypothesis generation, not after.
This reduces token spend on garbage hypotheses.

Checks (from DESIGN.md):
1. Prerequisite check — are all required techniques/data in the graph?
2. Contradiction check — does the graph contain counter-evidence?
3. "Already tried" check — did papers cite both endpoints without combining?
4. Implementation constraint check — mathematical/hardware incompatibilities
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

from kosa.activation.typed_walk import GraphInterface, WalkPath
from kosa.graph.schema import EdgeType

logger = logging.getLogger(__name__)


class FalsificationReason(StrEnum):
    PASS = "pass"
    MISSING_PREREQUISITE = "missing_prerequisite"
    CONTRADICTION = "contradiction"
    ALREADY_TRIED = "already_tried"
    INCOMPATIBLE = "incompatible"


@dataclass
class FalsificationCheck:
    """Result of a single falsification check."""

    check_type: str
    passed: bool
    reason: str = ""
    evidence: list[str] = field(default_factory=list)


@dataclass
class FalsificationResult:
    """Full falsification result for a candidate path."""

    path: WalkPath
    overall: FalsificationReason = FalsificationReason.PASS
    checks: list[FalsificationCheck] = field(default_factory=list)

    @property
    def survived(self) -> bool:
        return self.overall == FalsificationReason.PASS

    def summary(self) -> str:
        status = "PASS" if self.survived else f"FAIL ({self.overall.value})"
        failed = [c for c in self.checks if not c.passed]
        if failed:
            reasons = "; ".join(c.reason for c in failed)
            return f"{status}: {reasons}"
        return status


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_prerequisites(
    path: WalkPath,
    graph: GraphInterface,
) -> FalsificationCheck:
    """Check if all prerequisite techniques exist in the graph.

    For a path A → B, check that A's prerequisites (USES edges) all exist.
    If the path proposes combining technique A with problem B, the techniques
    that A depends on must be present in the graph.
    """
    missing = []

    for node in path.nodes:
        if node.node_type != "Technique":
            continue

        neighbors = graph.get_neighbors(node.node_id)
        uses_edges = [n for n in neighbors if n.edge_type == EdgeType.USES]

        for dep in uses_edges:
            dep_node = graph.get_node(dep.node_id)
            if dep_node is None:
                missing.append(f"{node.node_name} requires {dep.node_name} (not in graph)")

    if missing:
        return FalsificationCheck(
            check_type="prerequisite",
            passed=False,
            reason=f"Missing prerequisites: {', '.join(missing[:3])}",
            evidence=missing,
        )

    return FalsificationCheck(
        check_type="prerequisite",
        passed=True,
        reason="All prerequisites present",
    )


def check_contradictions(
    path: WalkPath,
    graph: GraphInterface,
) -> FalsificationCheck:
    """Check if the graph contains edges contradicting this path.

    Looks for:
    - Technique at start has HAS_LIMITATION pointing to the same problem
      the path claims it MITIGATES (contradictory claims)
    - Two techniques in the path have IMPROVES_OVER edges suggesting
      one supersedes the other
    """
    contradictions = []

    # Collect all technique and problem nodes in path
    techniques = [n for n in path.nodes if n.node_type == "Technique"]
    problems = [n for n in path.nodes if n.node_type == "Problem"]

    for tech in techniques:
        neighbors = graph.get_neighbors(tech.node_id)

        for problem in problems:
            # Check if this technique has a limitation that IS the problem
            # it's supposed to mitigate
            limitations = [
                n
                for n in neighbors
                if n.edge_type == EdgeType.HAS_LIMITATION and n.node_id == problem.node_id
            ]
            mitigations = [e for e in path.edges if e.edge_type == EdgeType.MITIGATES]

            if limitations and mitigations:
                contradictions.append(
                    f"{tech.node_name} has limitation "
                    f"'{problem.node_name}' but path claims mitigation"
                )

    # Check for supersession: if A IMPROVES_OVER B and both are in path
    for i, tech_a in enumerate(techniques):
        neighbors_a = graph.get_neighbors(tech_a.node_id)
        for tech_b in techniques[i + 1 :]:
            improves = [
                n
                for n in neighbors_a
                if n.edge_type == EdgeType.IMPROVES_OVER and n.node_id == tech_b.node_id
            ]
            if improves:
                contradictions.append(
                    f"{tech_a.node_name} already improves over "
                    f"{tech_b.node_name} — combination may be redundant"
                )

    if contradictions:
        return FalsificationCheck(
            check_type="contradiction",
            passed=False,
            reason=contradictions[0],
            evidence=contradictions,
        )

    return FalsificationCheck(
        check_type="contradiction",
        passed=True,
        reason="No contradictions found",
    )


def check_already_tried(
    path: WalkPath,
    graph: GraphInterface,
) -> FalsificationCheck:
    """Check if papers cite both endpoints but don't combine them.

    If paper C cites both technique A and technique B (the endpoints),
    but there's no edge between A and B, it may indicate the combination
    was considered and rejected.
    """
    if len(path.nodes) < 2:
        return FalsificationCheck(
            check_type="already_tried",
            passed=True,
            reason="Path too short to check",
        )

    start = path.nodes[0]
    end = path.nodes[-1]

    # Only meaningful for technique/problem nodes
    if start.node_type == "Paper" or end.node_type == "Paper":
        return FalsificationCheck(
            check_type="already_tried",
            passed=True,
            reason="Endpoints are papers — no combination to check",
        )

    # Find papers connected to start node
    start_neighbors = graph.get_neighbors(start.node_id)
    start_papers = {
        n.node_id
        for n in start_neighbors
        if n.node_type == "Paper" or n.edge_type in {EdgeType.INTRODUCES, EdgeType.EVALUATES_ON}
    }

    # Find papers connected to end node
    end_neighbors = graph.get_neighbors(end.node_id)
    end_papers = {
        n.node_id
        for n in end_neighbors
        if n.node_type == "Paper" or n.edge_type in {EdgeType.INTRODUCES, EdgeType.EVALUATES_ON}
    }

    # Papers that know about both
    shared_papers = start_papers & end_papers

    if shared_papers:
        # Check if there's already a direct edge between start and end
        has_direct_edge = any(n.node_id == end.node_id for n in start_neighbors)

        if not has_direct_edge:
            return FalsificationCheck(
                check_type="already_tried",
                passed=False,
                reason=(
                    f"{len(shared_papers)} paper(s) cite both "
                    f"'{start.node_name}' and '{end.node_name}' "
                    f"but they're not connected — may have been "
                    f"considered and rejected"
                ),
                evidence=[f"Shared papers: {list(shared_papers)[:5]}"],
            )

    return FalsificationCheck(
        check_type="already_tried",
        passed=True,
        reason="No evidence of prior combination attempt",
    )


def check_implementation_constraints(
    path: WalkPath,
    graph: GraphInterface,
) -> FalsificationCheck:
    """Check for hardware, data format, or mathematical incompatibilities.

    Looks for:
    - Mathematical structure mismatches between techniques
    - Known incompatibilities encoded as HAS_LIMITATION edges
    """
    # For Phase 1, this is a lightweight check based on node properties.
    # More sophisticated checks (hardware requirements, data format
    # compatibility) require richer node properties added in Phase 1.5+.
    issues = []

    techniques = [n for n in path.nodes if n.node_type == "Technique"]

    # Check if any technique has a limitation related to another technique
    for tech in techniques:
        neighbors = graph.get_neighbors(tech.node_id)
        limitations = [n for n in neighbors if n.edge_type == EdgeType.HAS_LIMITATION]

        for other_tech in techniques:
            if other_tech.node_id == tech.node_id:
                continue
            # Check if any limitation mentions the other technique's domain
            for lim in limitations:
                lim_name = lim.node_name.lower()
                other_name = other_tech.node_name.lower()
                if other_name in lim_name:
                    issues.append(
                        f"{tech.node_name} has limitation related to "
                        f"{other_tech.node_name}: {lim.node_name}"
                    )

    if issues:
        return FalsificationCheck(
            check_type="implementation_constraint",
            passed=False,
            reason=issues[0],
            evidence=issues,
        )

    return FalsificationCheck(
        check_type="implementation_constraint",
        passed=True,
        reason="No implementation constraints detected",
    )


# ---------------------------------------------------------------------------
# Main falsification pipeline
# ---------------------------------------------------------------------------


def falsify_path(
    path: WalkPath,
    graph: GraphInterface,
) -> FalsificationResult:
    """Run all falsification checks on a candidate path.

    Returns FalsificationResult with overall pass/fail and individual checks.
    Stops at the first failure (fail-fast).
    """
    result = FalsificationResult(path=path)

    checks = [
        ("prerequisite", check_prerequisites),
        ("contradiction", check_contradictions),
        ("already_tried", check_already_tried),
        ("implementation_constraint", check_implementation_constraints),
    ]

    for check_name, check_fn in checks:
        check_result = check_fn(path, graph)
        result.checks.append(check_result)

        if not check_result.passed:
            # Map check name to falsification reason
            reason_map = {
                "prerequisite": FalsificationReason.MISSING_PREREQUISITE,
                "contradiction": FalsificationReason.CONTRADICTION,
                "already_tried": FalsificationReason.ALREADY_TRIED,
                "implementation_constraint": FalsificationReason.INCOMPATIBLE,
            }
            result.overall = reason_map.get(check_name, FalsificationReason.CONTRADICTION)
            logger.debug(f"Path falsified ({check_name}): {check_result.reason}")
            break  # Fail fast

    return result


def falsify_paths(
    paths: list[WalkPath],
    graph: GraphInterface,
) -> tuple[list[WalkPath], list[FalsificationResult]]:
    """Falsify multiple paths, returning survivors and all results.

    Returns:
        (surviving_paths, all_results) — paths that passed all checks,
        and full results for analysis.
    """
    survivors = []
    all_results = []

    for path in paths:
        result = falsify_path(path, graph)
        all_results.append(result)
        if result.survived:
            survivors.append(path)

    total = len(paths)
    killed = total - len(survivors)
    logger.info(f"Falsification: {len(survivors)}/{total} paths survived ({killed} falsified)")

    return survivors, all_results
