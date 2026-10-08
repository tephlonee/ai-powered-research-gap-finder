"""
The extraction schema — the single source of truth for what we pull from each paper.

Design (from the first-principles rebuild):
  * STATIC CORE      — these fields are domain-general and stable. They are, at once,
                       the COLUMNS of a human literature-review synthesis matrix AND
                       the NODE/EDGE types of the knowledge graph. Extract once, use twice.
  * OPEN VOCABULARY  — the *values* (names of methods, components, datasets, …) are
                       discovered from the corpus, never hardcoded. Nothing here says
                       "roller bearing"; the system learns that from papers.
  * VERSIONED        — bump config.schema_version to force re-extraction. The cache key
                       includes it, so a schema change is a real migration: only the
                       affected computations are redone.

Compared with the old OWL ontology this replaces: no domain lock-in (no RLController /
AssemblyLine classes), no Java/HermiT reasoner, and the slots a bearing-PdM paper
actually needs — Dataset, Metric, Assumption, Baseline, Component, FailureMode — now exist.

NOTE ON OPTIONAL FIELDS: nested optional fields default to "" rather than Optional[...]
on purpose. Ollama's schema-constrained decoding is most reliable on simple schemas
(no anyOf/null unions), so a smaller local model still emits valid JSON every time.
"""
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    TASK = "Task"
    METHOD = "Method"
    COMPONENT = "Component"       # the thing a method operates on (bearing, cell, image domain)
    DATASET = "Dataset"
    METRIC = "Metric"
    ASSUMPTION = "Assumption"
    BASELINE = "Baseline"
    FAILURE_MODE = "FailureMode"


class Entity(BaseModel):
    """A typed entity with its provenance sentence, so every graph node is traceable."""
    name: str = Field(..., description="Surface form exactly as written in the paper.")
    type: EntityType = Field(..., description="Which core slot this entity fills.")
    evidence: str = Field("", description="The sentence/span it was taken from.")


class Result(BaseModel):
    """
    A reified n-ary fact: a method, optionally on a component/dataset, achieving a
    metric value, under assumptions. Reification (a Result node rather than loose
    pairwise edges) is what later makes contradiction detection and evaluation-gap
    detection possible — you can tell which score went with which dataset.
    """
    method: str = Field("", description="Method this result is about.")
    component: str = Field("", description="Component/system it was applied to, if stated.")
    dataset: str = Field("", description="Dataset it was evaluated on, if stated.")
    metric: str = Field("", description="Metric name, if stated.")
    value: str = Field("", description="Metric value as written, if stated.")
    evidence: str = Field("", description="Sentence this result came from.")


class PaperExtraction(BaseModel):
    """Everything we pull from one paper. Lists hold canonical-pending surface forms."""
    tasks: list[str] = Field(default_factory=list, description="Problems/tasks addressed.")
    methods: list[str] = Field(default_factory=list, description="Methods/models/approaches used.")
    components: list[str] = Field(default_factory=list, description="Things the methods operate on.")
    datasets: list[str] = Field(default_factory=list, description="Datasets/benchmarks used.")
    metrics: list[str] = Field(default_factory=list, description="Evaluation metrics used.")
    baselines: list[str] = Field(default_factory=list, description="Baselines compared against.")
    assumptions: list[str] = Field(default_factory=list, description="Stated or clearly-implied assumptions.")
    failure_modes: list[str] = Field(default_factory=list, description="Failure modes/fault types studied.")
    limitations: list[str] = Field(default_factory=list, description="Stated limitations (verbatim-ish).")
    future_work: list[str] = Field(default_factory=list, description="Stated future work / open problems.")
    results: list[Result] = Field(default_factory=list, description="Reified method/dataset/metric facts.")
    entities: list[Entity] = Field(default_factory=list, description="All typed entities with provenance.")


# --- Graph relationship vocabulary ------------------------------------------
# A FIXED, closed set of relationship types. The old code interpolated LLM-produced
# strings straight into Cypher as relationship/label names — a Cypher-injection and
# label-explosion risk. Here the LLM only ever fills node *values*; edge *types* come
# from this enum, never from model output.
class RelType(str, Enum):
    MENTIONS = "MENTIONS"              # (Paper)->(Entity)
    APPLIED_TO = "APPLIED_TO"          # (Method)->(Component)
    EVALUATED_ON = "EVALUATED_ON"      # (Method)->(Dataset)
    ACHIEVES = "ACHIEVES"              # (Method)->(Metric)
    ASSUMES = "ASSUMES"                # (Method)->(Assumption)
    HAS_FAILURE_MODE = "HAS_FAILURE_MODE"  # (Component)->(FailureMode)
    CITES = "CITES"                    # (Paper)->(Paper)
    REPORTS = "REPORTS"                # (Paper)->(Result)
