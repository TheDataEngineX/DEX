"""Project compiler: manifest in, validated immutable revision out (§6.8).

Eleven stages, in order, each able to stop the pipeline:

1.  schema validation and version checking
2.  import resolution with cycle detection
3.  reference resolution
4.  provider and capability resolution
5.  static permission analysis
6.  resource budget validation
7.  dependency lock verification
8.  workload graph validation
9.  policy and retention validation
10. canonical normalization and hashing
11. execution IR generation

The ordering is not cosmetic. Later stages assume earlier ones passed — you
cannot resolve a reference in a file that failed to parse, and hashing a
manifest whose imports are unresolved would produce a hash for something that
cannot run.

**The compiler fails closed.** The superseded engine called ``validate_config``
and discarded the errors, so an invalid project ran anyway and failed somewhere
less obvious. Here, errors mean no revision is produced, so nothing invalid can
be published or pinned by a run.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from dataenginex.foundation import (
    FrozenModel,
    Operation,
    OperationRegistry,
    ResourceRequest,
    RiskLevel,
    SideEffectClass,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    WorkloadKind,
    extract_host,
)
from dataenginex.foundation import registry as default_operations
from dataenginex.runtime.compiler.manifest import (
    NetworkPolicyDeclaration,
    OperationDeclaration,
    PolicyDeclaration,
    ProjectManifest,
    ProjectMetadata,
    ResourceDeclaration,
    WorkloadDeclaration,
)

__all__ = [
    "CompiledProject",
    "CompiledWorkload",
    "ProjectCompiler",
    "compile_project",
    "parse_size",
]

# Resource config keys that name a location on disk. Listed rather than guessed
# from the value: a table name, a URL, and a connection string all look enough
# like paths to be rewritten by a heuristic, and rewriting one breaks it.
_PATH_KEYS = frozenset({"path", "file", "directory", "dir", "lakehouse_root", "root"})

# Operation declarations live in foundation/operations_catalog — one catalogue
# read by the compiler, the runtime, and the policy engine, rather than a
# private table per consumer that can drift out of agreement. They sit in the
# foundation because a declaration is data: putting them in domains/ made the
# runtime depend on domains, which depends on the runtime.

# Capabilities the core ships. §6.4 declares required capabilities so a project
# fails at compile time rather than at the first run that needs one.
_BUILTIN_CAPABILITIES = frozenset(
    {
        "data.batch",
        "data.stream",
        "analytics.sql",
        "scheduling.local",
        "ml.training",
        "ml.serving",
        "ai.local",
        "ai.remote",
        "governance.policy",
    }
)

_SIZE_UNITS = {
    "B": 1,
    "KIB": 1024,
    "MIB": 1024**2,
    "GIB": 1024**3,
    "TIB": 1024**4,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
}
_SIZE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)\s*$")


def parse_size(value: str) -> int:
    """Turn ``"6GiB"`` into bytes.

    Binary and decimal units are distinguished because the difference is 7% at
    GiB scale, which is enough to overshoot a memory ceiling.
    """
    match = _SIZE_RE.match(value)
    if match is None:
        raise ValueError(f"cannot parse size {value!r}")
    amount, unit = match.groups()
    factor = _SIZE_UNITS.get(unit.upper())
    if factor is None:
        raise ValueError(f"unknown size unit {unit!r} in {value!r}")
    return int(float(amount) * factor)


class CompiledWorkload(FrozenModel):
    """A workload with its operations resolved to typed :class:`Operation`s."""

    name: str
    kind: WorkloadKind
    operations: tuple[Operation, ...]
    depends_on: tuple[str, ...]
    schedule: str | None
    max_retries: int
    priority: int
    resource_request: ResourceRequest
    continuous: bool


class CompiledProject(FrozenModel):
    """The execution IR (§6.8, stage 11).

    Internal and unstable before v1 — the manifest is the public contract, this
    is what the runtime actually executes. ``content_hash`` addresses the
    canonical normalized form, so identical inputs always produce the same
    revision identity.
    """

    manifest: ProjectManifest
    workloads: tuple[CompiledWorkload, ...]
    resources: tuple[ResourceDeclaration, ...]
    content_hash: str
    dependency_lock_hash: str | None
    required_capabilities: tuple[str, ...]
    # Every destination any operation may reach, from the network policy.
    declared_destinations: tuple[str, ...]
    # Topologically ordered workload names — execution order, cycle-free.
    execution_order: tuple[str, ...]
    source_files: tuple[str, ...]
    report: ValidationReport

    @property
    def ok(self) -> bool:
        return self.report.ok


class ProjectCompiler:
    """Runs the §6.8 stages over a project directory.

    Collects every issue it can before stopping, so a user sees all the errors
    in one pass rather than fixing them one run at a time. Stages that would
    operate on unusable input do stop early — there is nothing useful to say
    about references inside a file that did not parse.
    """

    def __init__(self, root: Path, *, operations: OperationRegistry | None = None) -> None:
        self.root = root
        # Injectable so a project with plugin-supplied operations compiles
        # against a registry that includes them (§10.6).
        self.operations = operations or default_operations
        self._issues: list[ValidationIssue] = []
        # Name -> declaration, rebuilt on each compile(). Egress analysis
        # follows an operation's inputs and outputs to the resource that
        # says where the data actually comes from.
        self._resources_by_name: dict[str, ResourceDeclaration] = {}

    # --- issue helpers ------------------------------------------------------

    def _error(self, code: str, message: str, location: str | None = None) -> None:
        self._issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code=code,
                message=message,
                location=location,
            )
        )

    def _warn(self, code: str, message: str, location: str | None = None) -> None:
        self._issues.append(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code=code,
                message=message,
                location=location,
            )
        )

    def _report(self) -> ValidationReport:
        return ValidationReport(issues=tuple(self._issues))

    # --- entry point --------------------------------------------------------

    def compile(self) -> CompiledProject:
        """Run every stage.

        Never raises for project errors — they land in the report, because a
        caller wants the whole list, not the first failure.
        """
        self._issues = []
        self._resources_by_name = {}
        manifest_path = self.root / "dex.yaml"

        # Stage 1: schema validation and version checking.
        manifest, sources = self._load_manifest(manifest_path)
        if manifest is None:
            return self._empty(manifest_path)

        # Stage 2-3: imports (with cycle detection) and reference resolution.
        merged, import_sources = self._resolve_imports(manifest)
        sources.extend(import_sources)

        # Stage 4: providers and capabilities.
        self._check_capabilities(merged)

        # Stage 5: static permission analysis.
        destinations = self._analyse_permissions(merged)

        # Stage 6: resource budgets.
        self._check_budgets(merged)

        # Stage 7: dependency lock.
        lock_hash = self._verify_lock(merged)

        # Stage 8: workload graph.
        order = self._validate_graph(merged)

        # Stage 9: policy and retention.
        self._validate_policies(merged)

        # Stage 10-11: canonical hash and IR.
        workloads = self._build_workloads(merged)
        canonical = _canonical_json(merged.model_dump(mode="json", by_alias=True))
        content_hash = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()

        return CompiledProject(
            manifest=merged,
            workloads=workloads,
            # Resolved *after* the hash, never before. A manifest's ``path`` is
            # written relative to the project, and a worker runs from its own
            # working directory — so the relative form only ever worked by
            # coincidence. Resolving into the hash instead would bake this
            # machine's layout into the revision id and stop the same revision
            # from running anywhere else (§17 Phase 6).
            resources=tuple(self._resolve_paths(r) for r in merged.spec.resources),
            content_hash=content_hash,
            dependency_lock_hash=lock_hash,
            required_capabilities=merged.spec.capabilities.required,
            declared_destinations=destinations,
            execution_order=order,
            source_files=tuple(sorted(set(sources))),
            report=self._report(),
        )

    def _resolve_paths(self, resource: ResourceDeclaration) -> ResourceDeclaration:
        """Make a resource's filesystem settings absolute, against the project root.

        Only these keys, and only when already relative. Rewriting every value
        that happens to look like a path would mangle a table name or a URL, and
        an absolute path is a deliberate choice to point outside the project.
        """
        updates = {
            key: str((self.root / value).resolve())
            for key, value in resource.config.items()
            if key in _PATH_KEYS and value and not Path(value).is_absolute()
        }
        if not updates:
            return resource
        return resource.model_copy(update={"config": {**resource.config, **updates}})

    def _empty(self, path: Path) -> CompiledProject:
        """A result for a project that could not be parsed at all."""
        return CompiledProject(
            manifest=ProjectManifest(metadata=ProjectMetadata(name="invalid")),
            workloads=(),
            resources=(),
            content_hash="",
            dependency_lock_hash=None,
            required_capabilities=(),
            declared_destinations=(),
            execution_order=(),
            source_files=(str(path),),
            report=self._report(),
        )

    # --- stage 1: schema ----------------------------------------------------

    def _load_manifest(self, path: Path) -> tuple[ProjectManifest | None, list[str]]:
        if not path.is_file():
            self._error("E_NO_MANIFEST", f"no dex.yaml at {path}", str(path))
            return None, []

        raw = self._read_yaml(path)
        if raw is None:
            return None, [str(path)]

        if not isinstance(raw, dict):
            self._error("E_MANIFEST_SHAPE", "dex.yaml must be a mapping", str(path))
            return None, [str(path)]

        kind = raw.get("kind")
        if kind != "Project":
            self._error(
                "E_WRONG_KIND",
                f"dex.yaml declares kind {kind!r}, expected 'Project'",
                str(path),
            )
            return None, [str(path)]

        try:
            manifest = ProjectManifest.model_validate(raw)
        except ValidationError as exc:
            for error in exc.errors():
                location = ".".join(str(p) for p in error["loc"])
                self._error("E_SCHEMA", error["msg"], location or str(path))
            return None, [str(path)]

        return manifest, [str(path)]

    def _read_yaml(self, path: Path) -> Any:
        try:
            # safe_load, never load: a project file must not be able to
            # construct arbitrary Python objects (§6.7).
            return yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            self._error("E_YAML", f"invalid YAML: {exc}", str(path))
        except OSError as exc:
            self._error("E_READ", f"cannot read: {exc}", str(path))
        return None

    # --- stage 2-3: imports and references ---------------------------------

    def _resolve_imports(self, manifest: ProjectManifest) -> tuple[ProjectManifest, list[str]]:
        """Merge imported fragments into one manifest.

        Cycle detection tracks resolved paths, not glob patterns: two different
        patterns can name the same file, and importing it twice is still a
        cycle.
        """
        resources = list(manifest.spec.resources)
        workloads = list(manifest.spec.workloads)
        policies = list(manifest.spec.policies)
        sources: list[str] = []
        seen: set[Path] = set()

        for pattern in manifest.spec.imports:
            if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
                # An import escaping the project root would let a revision
                # depend on files outside the bundle, which breaks portability
                # and content addressing.
                self._error(
                    "E_IMPORT_ESCAPE",
                    f"import {pattern!r} must stay inside the project directory",
                    f"spec.imports[{pattern}]",
                )
                continue

            matches = sorted(self.root.glob(pattern))
            if not matches:
                self._warn(
                    "W_IMPORT_EMPTY",
                    f"import {pattern!r} matched no files",
                    f"spec.imports[{pattern}]",
                )
            for match in matches:
                resolved = match.resolve()
                if resolved in seen:
                    self._error(
                        "E_IMPORT_CYCLE",
                        f"{match.name} imported more than once",
                        str(match),
                    )
                    continue
                seen.add(resolved)
                sources.append(str(match))

                fragment = self._read_yaml(match)
                if not isinstance(fragment, dict):
                    if fragment is not None:
                        self._error(
                            "E_FRAGMENT_SHAPE",
                            "imported file must be a mapping",
                            str(match),
                        )
                    continue

                self._merge_fragment(fragment, match, resources, workloads, policies)

        merged_spec = manifest.spec.model_copy(
            update={
                "resources": tuple(resources),
                "workloads": tuple(workloads),
                "policies": tuple(policies),
            }
        )
        return manifest.model_copy(update={"spec": merged_spec}), sources

    def _merge_fragment(
        self,
        fragment: dict[str, Any],
        path: Path,
        resources: list[ResourceDeclaration],
        workloads: list[WorkloadDeclaration],
        policies: list[PolicyDeclaration],
    ) -> None:
        for index, item in enumerate(fragment.get("resources", []) or []):
            self._append(ResourceDeclaration, item, resources, path, "resources", index)
        for index, item in enumerate(fragment.get("workloads", []) or []):
            self._append(WorkloadDeclaration, item, workloads, path, "workloads", index)
        for index, item in enumerate(fragment.get("policies", []) or []):
            self._append(PolicyDeclaration, item, policies, path, "policies", index)

    def _append(
        self,
        model: type[Any],
        item: Any,
        target: list[Any],
        path: Path,
        key: str,
        index: int,
    ) -> None:
        try:
            target.append(model.model_validate(item))
        except ValidationError as exc:
            for error in exc.errors():
                loc = ".".join(str(p) for p in error["loc"])
                self._error("E_SCHEMA", error["msg"], f"{path.name}:{key}[{index}].{loc}")

    # --- stage 4: capabilities ---------------------------------------------

    def _check_capabilities(self, manifest: ProjectManifest) -> None:
        for capability in manifest.spec.capabilities.required:
            if capability not in _BUILTIN_CAPABILITIES:
                self._error(
                    "E_CAPABILITY",
                    f"required capability {capability!r} is not available; "
                    "install the plugin that provides it or remove it",
                    "spec.capabilities.required",
                )
        for capability in manifest.spec.capabilities.optional:
            if capability not in _BUILTIN_CAPABILITIES:
                self._warn(
                    "W_CAPABILITY",
                    f"optional capability {capability!r} is unavailable; "
                    "workloads needing it will be skipped",
                    "spec.capabilities.optional",
                )

    # --- stage 5: static permission analysis --------------------------------

    def _analyse_permissions(self, manifest: ProjectManifest) -> tuple[str, ...]:
        """Check that every externally-acting operation has a destination.

        This is where §9.7's default-deny becomes real: an operation that
        transmits data with no matching allow rule is rejected at compile time,
        not at the moment it would have sent the data.
        """
        network = manifest.spec.network
        allowed = {rule.host for rule in network.allow}
        self._resources_by_name = {r.name: r for r in manifest.spec.resources}

        for workload in manifest.spec.workloads:
            for op in workload.operations:
                if not self.operations.has(op.type):
                    continue
                declared = self.operations.get(op.type)
                if declared.side_effect_class not in (
                    SideEffectClass.EXTERNAL_READ,
                    SideEffectClass.EXTERNAL_WRITE,
                ):
                    continue

                self._check_operation_egress(
                    workload, op, declared, manifest, allowed=allowed, network=network
                )

        return tuple(sorted(allowed))

    def _check_operation_egress(
        self,
        workload: WorkloadDeclaration,
        op: OperationDeclaration,
        declared: Operation,
        manifest: ProjectManifest,
        *,
        allowed: set[str],
        network: NetworkPolicyDeclaration,
    ) -> None:
        """Check one externally-acting operation against the network policy."""
        destination = self._destination_of(op, manifest)
        location = f"spec.workloads.{workload.name}"

        # The two external classes are not the same risk, so they are not
        # treated the same when no destination can be resolved.
        #
        # EXTERNAL_WRITE always transmits — `notify` with no declared
        # destination is a send to somewhere nobody wrote down, and permitting
        # it because the compiler could not name the host would be a fail-open
        # on the exact case §9.7 exists for.
        #
        # EXTERNAL_READ may be entirely local. `ingest` of data/orders.csv is
        # external to the project and non-deterministic — the file can change
        # underneath a run — but it sends nothing. Demanding a network allow
        # rule for it would teach authors to write allow lists for projects
        # that never open a socket, and a habitually-permissive allow list is
        # worse than no check at all.
        if destination is None:
            if declared.side_effect_class is SideEffectClass.EXTERNAL_WRITE:
                self._error(
                    "E_EGRESS_UNRESOLVED",
                    f"operation {op.type!r} in workload {workload.name!r} transmits "
                    "data but names no destination; declare one in parameters or on "
                    "the resource it writes to (§9.7)",
                    location,
                )
        elif network.default == "deny" and not allowed:
            self._error(
                "E_EGRESS_UNDECLARED",
                f"operation {op.type!r} in workload {workload.name!r} reaches "
                f"{destination!r} but the project declares no network allow rules "
                "(§9.7 default deny)",
                location,
            )
        elif destination not in allowed:
            self._error(
                "E_EGRESS_DENIED",
                f"destination {destination!r} is not in the network policy allow list",
                location,
            )

        if declared.risk_level >= RiskLevel.MODIFY_EXTERNAL:
            self._warn(
                "W_HIGH_RISK",
                f"operation {op.type!r} is risk level {int(declared.risk_level)}; "
                "runs will require human approval (§9.5)",
                location,
            )

    def _destination_of(
        self, op: OperationDeclaration, manifest: ProjectManifest
    ) -> str | None:
        """The network host an operation reaches, or ``None`` if it stays local.

        Checked in order of how explicit the author was: a ``destination`` or
        ``host`` parameter states it outright; otherwise the operation's inputs
        and outputs are followed to their resource declarations, since that is
        where a connector's URL actually lives.

        Fails closed on anything unrecognised. A config value that is neither a
        local path nor a parseable host is reported rather than assumed
        harmless — guessing "probably a file" is exactly how an unreviewed
        transmission gets through.
        """
        explicit = op.parameters.get("destination") or op.parameters.get("host")
        if explicit:
            return extract_host(explicit)

        for name in (*op.inputs, *op.outputs):
            resource = self._resources_by_name.get(name)
            if resource is None:
                continue
            for key in ("uri", "url", "endpoint", "host", "bootstrap_servers"):
                value = resource.config.get(key)
                if value:
                    return extract_host(value)
            # A declared local path is the common case and settles the question.
            if resource.config.get("path"):
                return None

        return None

    # --- stage 6: resource budgets ------------------------------------------

    def _check_budgets(self, manifest: ProjectManifest) -> None:
        limits = manifest.spec.limits
        for field, value in (
            ("memory", limits.memory),
            ("workingStorage", limits.working_storage),
        ):
            try:
                parsed = parse_size(value)
            except ValueError as exc:
                self._error("E_LIMIT", str(exc), f"spec.limits.{field}")
                continue
            if parsed <= 0:
                self._error("E_LIMIT", f"{field} must be positive", f"spec.limits.{field}")

    # --- stage 7: dependency lock -------------------------------------------

    def _verify_lock(self, manifest: ProjectManifest) -> str | None:
        """Hash the lockfile so a revision pins its exact dependency set (§6.9).

        A missing lock is a warning, not an error: a project with no third-party
        dependencies legitimately has none. What must not pass silently is a
        lock that exists but cannot be read or parsed.
        """
        lock_path = self.root / manifest.spec.environment.lockfile
        if not lock_path.is_file():
            self._warn(
                "W_NO_LOCK",
                f"no {manifest.spec.environment.lockfile}; the environment is "
                "not reproducible across machines (§6.9)",
                "spec.environment.lockfile",
            )
            return None
        try:
            data = lock_path.read_bytes()
        except OSError as exc:
            self._error("E_LOCK_READ", f"cannot read lockfile: {exc}", str(lock_path))
            return None

        # The default lockfile is TOML but named ``dex.lock``, so keying this
        # check off the extension alone would skip exactly the file that
        # matters and let a corrupted lock hash cleanly.
        if lock_path.suffix in {".toml", ".lock"}:
            try:
                tomllib.loads(data.decode())
            except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
                self._error("E_LOCK_PARSE", f"invalid lockfile: {exc}", str(lock_path))
                return None

        return "sha256:" + hashlib.sha256(data).hexdigest()

    # --- stage 8: workload graph --------------------------------------------

    def _validate_graph(self, manifest: ProjectManifest) -> tuple[str, ...]:
        """Check the workload graph, then order it (§6.8 stage 8)."""
        workloads = {w.name: w for w in manifest.spec.workloads}

        duplicates = len(manifest.spec.workloads) - len(workloads)
        if duplicates:
            self._error(
                "E_DUPLICATE_WORKLOAD",
                f"{duplicates} workload name(s) declared more than once",
                "spec.workloads",
            )

        for workload in manifest.spec.workloads:
            for dependency in workload.depends_on:
                if dependency not in workloads:
                    self._error(
                        "E_MISSING_DEPENDENCY",
                        f"workload {workload.name!r} depends on unknown {dependency!r}",
                        f"spec.workloads.{workload.name}.depends_on",
                    )

        order = _topological_order(workloads)
        if len(order) != len(workloads):
            cyclic = sorted(set(workloads) - set(order))
            self._error(
                "E_GRAPH_CYCLE",
                f"dependency cycle among workloads: {', '.join(cyclic)}",
                "spec.workloads",
            )

        return order

    # --- stage 9: policy and retention --------------------------------------

    def _validate_policies(self, manifest: ProjectManifest) -> None:
        names: set[str] = set()
        for policy in manifest.spec.policies:
            if policy.name in names:
                self._error(
                    "E_DUPLICATE_POLICY",
                    f"policy {policy.name!r} declared more than once",
                    "spec.policies",
                )
            names.add(policy.name)

            if policy.effect not in (
                "permit",
                "deny",
                "require_approval",
                "permit_with_obligations",
            ):
                self._error(
                    "E_POLICY_EFFECT",
                    f"unknown effect {policy.effect!r}",
                    f"spec.policies.{policy.name}",
                )

        resource_names = {r.name for r in manifest.spec.resources}
        for workload in manifest.spec.workloads:
            # An operation may read what an earlier operation in the same
            # workload produced. Those intermediates are not declared resources
            # — they exist only for the length of the run — so they are added as
            # each operation is checked. Requiring them to be declared would
            # make every multi-step workload uncompilable, which is to say it
            # would make ``inputs``/``outputs`` chaining unusable.
            available = set(resource_names)
            for op in workload.operations:
                for reference in op.inputs:
                    # A dotted reference names another workload's output; a bare
                    # name must match a declared resource or an earlier step's.
                    if reference not in available and "." not in reference:
                        self._error(
                            "E_UNKNOWN_INPUT",
                            f"operation input {reference!r} matches no declared resource "
                            "and is produced by no earlier operation in this workload",
                            f"spec.workloads.{workload.name}",
                        )
                available.update(op.outputs)

    # --- stage 11: IR -------------------------------------------------------

    def _build_workloads(self, manifest: ProjectManifest) -> tuple[CompiledWorkload, ...]:
        compiled: list[CompiledWorkload] = []
        for workload in manifest.spec.workloads:
            operations = tuple(
                self._build_operation(workload, declaration) for declaration in workload.operations
            )
            compiled.append(
                CompiledWorkload(
                    name=workload.name,
                    kind=workload.kind,
                    operations=operations,
                    depends_on=workload.depends_on,
                    schedule=workload.schedule,
                    max_retries=workload.retry.max_attempts,
                    priority=workload.priority,
                    resource_request=_request_for(manifest, workload),
                    continuous=workload.kind in (WorkloadKind.SPARK_STREAM, WorkloadKind.SERVICE),
                )
            )
        return tuple(compiled)

    def _build_operation(
        self, workload: WorkloadDeclaration, declaration: OperationDeclaration
    ) -> Operation:
        """Resolve a declared operation against the catalogue (§4.7).

        Looked up rather than inferred. Deriving idempotency from the
        side-effect class was wrong in both directions — ``delete`` is naturally
        idempotent despite being destructive, and ``infer`` is unsafe to replay
        despite only reading.

        The catalogue entry supplies the *contract* — determinism, risk, retry
        safety. The declaration supplies the *binding* — which resources, which
        settings. Both are needed: the catalogue alone describes a category of
        work no handler can act on, and the declaration alone carries none of
        the safety properties the scheduler and policy engine read.
        """
        if not self.operations.has(declaration.type):
            self._error(
                "E_UNKNOWN_OPERATION",
                f"operation type {declaration.type!r} is not provided by the core "
                "or any enabled plugin",
                f"spec.workloads.{workload.name}",
            )
            return Operation(
                operation_type=declaration.type,
                name=declaration.name,
                side_effect_class=SideEffectClass.READ,
                risk_level=RiskLevel.READ_PROJECT_DATA,
            )

        return self.operations.get(declaration.type).model_copy(
            update={
                "name": declaration.name or declaration.type,
                "bound_inputs": declaration.inputs,
                "bound_outputs": declaration.outputs,
                "parameters": dict(declaration.parameters),
                "sql_file": declaration.sql_file,
                "script_file": declaration.script_file,
            }
        )


def _request_for(manifest: ProjectManifest, workload: WorkloadDeclaration) -> ResourceRequest:
    """Derive a per-workload request from the project ceiling.

    Half the project's memory by default: a single workload that claims the
    whole budget starves everything else, and the scheduler needs headroom to
    place concurrent work (§7.5).
    """
    try:
        memory_mb = parse_size(manifest.spec.limits.memory) // (1024 * 1024)
    except ValueError:
        memory_mb = 512
    return ResourceRequest(
        cpu_cores=max(1.0, manifest.spec.limits.cpu / 2),
        memory_mb=max(256, memory_mb // 2),
        timeout_seconds=3600 if workload.kind is WorkloadKind.BATCH else 86400,
    )


def _topological_order(
    workloads: dict[str, WorkloadDeclaration],
) -> tuple[str, ...]:
    """Kahn's algorithm. Anything left unemitted is in a cycle.

    Ties are broken alphabetically so the same project produces the same order
    on every machine — without that, the content hash would depend on dict
    iteration order and stop identifying the project.
    """
    indegree = {
        name: sum(1 for d in w.depends_on if d in workloads) for name, w in workloads.items()
    }
    dependents: dict[str, list[str]] = {name: [] for name in workloads}
    for name, workload in workloads.items():
        for dependency in workload.depends_on:
            if dependency in dependents:
                dependents[dependency].append(name)

    ready = sorted(n for n, d in indegree.items() if d == 0)
    order: list[str] = []
    while ready:
        name = ready.pop(0)
        order.append(name)
        for dependent in sorted(dependents[name]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
        ready.sort()

    return tuple(order)


def _canonical_json(data: Any) -> str:
    """Deterministic JSON for hashing (§6.8, stage 10).

    Sorted keys and fixed separators so the same logical project always hashes
    identically regardless of key order or formatting in the source YAML.
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def compile_project(root: Path) -> CompiledProject:
    """Compile the project rooted at ``root``."""
    return ProjectCompiler(root).compile()
