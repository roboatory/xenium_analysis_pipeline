from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def annotate_clusters_with_llm(
    annotation_evidence_by_cluster: dict[str, list[str]],
    *,
    model: str = "llama3.1:8b",
    evidence_type: str = "marker_genes",
    host: str = "http://localhost:11434",
    temperature: float = 0.0,
    timeout_seconds: int = 120,
) -> dict[str, dict[str, Any]]:
    """Annotate clusters using a local Ollama-compatible LLM."""

    if not annotation_evidence_by_cluster:
        return {}

    is_marker_mode = evidence_type == "marker_genes"
    evidence_label = (
        "marker genes" if is_marker_mode else "neighboring cell-type composition"
    )
    system_instruction = (
        "You are a domain expert in prostate cancer spatial transcriptomics. "
        "Use maximally specific labels (lineage + subtype + state), keep "
        f"labels biologically plausible from {evidence_label}, and keep every "
        "cluster cell_type string globally unique. Return JSON only."
        if is_marker_mode
        else (
            "You are a domain expert in prostate cancer spatial transcriptomics. "
            "Given neighboring cell-type composition, assign spatial-domain "
            "microenvironment labels (niche/interface/transition zone), not raw "
            "single-cell-type labels. Keep every cluster cell_type string globally "
            "unique. Return JSON only."
        )
    )

    try:
        response = _ollama_chat(
            payload={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_instruction,
                    },
                    {
                        "role": "user",
                        "content": _build_annotation_prompt(
                            annotation_evidence_by_cluster,
                            evidence_type=evidence_type,
                        ),
                    },
                ],
                "format": _annotation_schema(),
                "stream": False,
                "temperature": temperature,
            },
            host=host,
            timeout_seconds=timeout_seconds,
        )
    except RuntimeError:
        return {
            str(cluster_id): {
                "cell_type": f"unknown_cluster_{cluster_id}",
                "confidence": 0.0,
                "rationale": "Local LLM service unavailable; using fallback label.",
            }
            for cluster_id in sorted(annotation_evidence_by_cluster.keys(), key=str)
        }

    raw_content = response.get("message", {}).get("content", "")
    try:
        return _parse_and_normalize(
            raw_content,
            annotation_evidence_by_cluster,
            evidence_type=evidence_type,
        )
    except (RuntimeError, ValueError):
        return {
            str(cluster_id): {
                "cell_type": f"unknown_cluster_{cluster_id}",
                "confidence": 0.0,
                "rationale": "LLM output was not valid structured JSON.",
            }
            for cluster_id in sorted(annotation_evidence_by_cluster.keys(), key=str)
        }


def _build_annotation_prompt(
    annotation_evidence_by_cluster: dict[str, list[str]],
    *,
    evidence_type: str = "marker_genes",
) -> str:
    """Build prompt with schema and per-cluster annotation evidence."""

    is_marker_mode = evidence_type == "marker_genes"
    evidence_label = (
        "marker genes" if is_marker_mode else "dominant neighboring cell types"
    )
    support_phrase = "marker-supported" if is_marker_mode else "composition-supported"

    if is_marker_mode:
        lines = [
            f"Annotate each cluster with one main cell type label from {evidence_label}.",
            "Be as specific as possible (lineage + subtype + functional state).",
            f"If clusters are similar, disambiguate using {support_phrase} states.",
            "Cell type labels must be globally unique across clusters.",
            "Treat labels as duplicates if they differ only by case, spacing, punctuation, or singular/plural form.",
            "Do not reuse any label string across clusters.",
            "When disambiguating, use biologically meaningful qualifiers (lineage/subtype/state), not generic numbering.",
            "Bad: 'T cell' and 'T-cell' (duplicate). Good: 'Cytotoxic T-cell effector state' and 'Exhausted T-cell state'.",
            "Before returning JSON, verify there are zero duplicate cell_type labels; if duplicates exist, rewrite and re-check.",
            "Return JSON using this schema only:",
            '{ "annotations": [{ "cluster_id": "0", "cell_type": "label", "confidence": 0.0, "rationale": "..." }] }',
            "Confidence must be a number in [0, 1].",
            f"Rationale should be 1-2 sentences with discriminating {evidence_label}.",
            "",
            f"Clusters and {evidence_label}:",
        ]
    else:
        lines = [
            "Annotate each spatial domain with one microenvironment label from neighboring cell-type composition.",
            "Do not output only a single raw cell type name; use niche/interface/transition-zone wording.",
            "Good label styles: 'Tumor-immune interface', 'Fibro-inflammatory stroma niche', 'Basal-luminal transition zone'.",
            f"If domains are similar, disambiguate with {support_phrase} context.",
            "Cell type labels must be globally unique across clusters.",
            "Treat labels as duplicates if they differ only by case, spacing, punctuation, or singular/plural form.",
            "Do not reuse any label string across clusters.",
            "When disambiguating, use ecological qualifiers (niche/interface/transition/perivascular/stromal-adjacent).",
            "Bad: 'Tumor-immune interface' and 'tumor immune interface' (duplicate). Good: 'Tumor-immune interface' and 'Perivascular tumor-immune niche'.",
            "Before returning JSON, verify there are zero duplicate cell_type labels; if duplicates exist, rewrite and re-check.",
            "Return JSON using this schema only:",
            '{ "annotations": [{ "cluster_id": "0", "cell_type": "label", "confidence": 0.0, "rationale": "..." }] }',
            "Confidence must be a number in [0, 1].",
            f"Rationale should be 1-2 sentences with discriminating {evidence_label}, including interactions/co-occurrence when possible.",
            "",
            f"Clusters and {evidence_label}:",
        ]
    for cluster_id, evidence_items in sorted(
        annotation_evidence_by_cluster.items(), key=lambda item: item[0]
    ):
        lines.append(f"- {cluster_id}: {', '.join(evidence_items)}")
    return "\n".join(lines)


def _ollama_chat(
    payload: dict[str, Any],
    *,
    host: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Call an Ollama-compatible /api/chat endpoint."""

    url = f"{host.rstrip('/')}/api/chat"
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except (HTTPError, URLError) as exc:
        msg = f"Failed to reach local LLM at {url}: {exc}"
        raise RuntimeError(msg) from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        msg = "Local LLM returned invalid JSON."
        raise RuntimeError(msg) from exc


def _parse_and_normalize(
    raw_content: str,
    annotation_evidence_by_cluster: dict[str, list[str]],
    *,
    evidence_type: str = "marker_genes",
) -> dict[str, dict[str, Any]]:
    """Parse model output, normalize fields, and fill missing clusters."""

    content = raw_content.strip()
    if not content:
        msg = "Local LLM returned empty content."
        raise ValueError(msg)

    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
        if content[:4].lower() == "json":
            content = content[4:].strip()

    candidates = [content]
    if "{" in content and "}" in content:
        start, end = content.find("{"), content.rfind("}")
        if end > start:
            candidates.append(content[start : end + 1])
    if "[" in content and "]" in content:
        start, end = content.find("["), content.rfind("]")
        if end > start:
            candidates.append(content[start : end + 1])

    parsed: Any | None = None
    for candidate in dict.fromkeys(candidates):
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if parsed is None:
        msg = "Could not parse JSON from LLM response."
        raise RuntimeError(msg)

    if isinstance(parsed, dict):
        for key in ("annotations", "clusters", "results", "data"):
            value = parsed.get(key)
            if isinstance(value, (dict, list)):
                parsed = value
                break

    raw_items: list[tuple[str, Any]] = []
    if isinstance(parsed, dict):
        raw_items = [(str(cluster_id), value) for cluster_id, value in parsed.items()]
    elif isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            cluster_id = (
                item.get("cluster_id")
                or item.get("cluster")
                or item.get("id")
                or item.get("group")
            )
            if cluster_id is not None:
                raw_items.append((str(cluster_id), item))
            elif len(item) == 1:
                ((cluster_id, value),) = item.items()
                raw_items.append((str(cluster_id), value))

    if not raw_items:
        msg = (
            "Annotation response must be a dictionary, or a list of objects with "
            "'cluster_id'/'cluster'/'id'/'group'."
        )
        raise ValueError(msg)

    normalized: dict[str, dict[str, Any]] = {}
    for cluster_id, value in raw_items:
        if isinstance(value, str):
            value = {"cell_type": value, "confidence": 0.0, "rationale": ""}
        if not isinstance(value, dict):
            continue

        cell_type = str(
            value.get("cell_type")
            or value.get("label")
            or value.get("type")
            or "unknown"
        ).strip()
        if evidence_type == "neighborhood_cell_types":
            cell_type = _normalize_microenvironment_label(
                cell_type,
                annotation_evidence_by_cluster.get(str(cluster_id), []),
            )
        rationale = str(
            value.get("rationale")
            or value.get("reason")
            or value.get("explanation")
            or ""
        ).strip()

        confidence_raw = value.get("confidence", value.get("score", 0.0))
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence > 1.0 and confidence <= 100.0:
            confidence = confidence / 100.0
        confidence = max(0.0, min(confidence, 1.0))

        normalized[cluster_id] = {
            "cell_type": cell_type or "unknown",
            "confidence": confidence,
            "rationale": rationale,
        }

    for cluster_id in sorted(annotation_evidence_by_cluster.keys(), key=str):
        cluster_key = str(cluster_id)
        if cluster_key not in normalized:
            normalized[cluster_key] = {
                "cell_type": f"unknown_cluster_{cluster_key}",
                "confidence": 0.0,
                "rationale": "Model did not return an annotation for this cluster.",
            }

    return normalized


def _normalize_microenvironment_label(label: str, evidence_items: list[str]) -> str:
    """Convert trivial neighborhood labels into microenvironment-style labels."""

    clean_label = label.strip()
    if not clean_label:
        return "unknown_microenvironment"

    normalized_label = clean_label.replace("_", " ").strip().lower()
    if normalized_label in {"unknown", "unassigned"}:
        return "unknown_microenvironment"

    evidence_tokens = {item.strip().lower() for item in evidence_items if item.strip()}
    if normalized_label in evidence_tokens:
        return f"{clean_label}-rich microenvironment"

    microenvironment_tokens = {
        "microenvironment",
        "niche",
        "interface",
        "zone",
        "region",
        "transition",
        "neighborhood",
        "compartment",
    }
    if any(token in normalized_label for token in microenvironment_tokens):
        return clean_label
    return f"{clean_label} microenvironment"


def _annotation_schema() -> dict[str, Any]:
    """Schema used by Ollama to enforce response structure."""

    return {
        "type": "object",
        "required": ["annotations"],
        "properties": {
            "annotations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["cluster_id", "cell_type", "confidence", "rationale"],
                    "properties": {
                        "cluster_id": {"type": "string"},
                        "cell_type": {"type": "string"},
                        "confidence": {"type": "number"},
                        "rationale": {"type": "string"},
                    },
                    "additionalProperties": True,
                },
            }
        },
        "additionalProperties": True,
    }
