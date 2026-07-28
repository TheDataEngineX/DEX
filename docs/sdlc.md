# SDLC Overview

**Software Development Lifecycle for DataEngineX — stages, artifacts, and quality gates.**

> **Quick Links:** [Lifecycle Stages](#lifecycle-stages) · [Development Workflow](#development-workflow-summary) · [Quality Gates](#4-verify-ci)

______________________________________________________________________

This document defines the software development lifecycle (SDLC) for DataEngineX, including the required stages, artifacts, and quality gates.

## Goals

- Ship deterministic builds with a single immutable version per release.
- Enforce quality gates (lint, test, type checks, security scan) on every PR.
- Maintain a complete audit trail for promotion across environments.

## Lifecycle Stages

### 1) Plan

**Entry**: Feature request, bug report, or operational task.

**Activities**:

- Define scope and acceptance criteria.
- Identify impacted services, APIs, or pipelines.
- Track work in GitHub Issues and GitHub Projects (milestones, priorities, owners).

**Exit**: Clear acceptance criteria and implementation plan.

**Artifacts**: GitHub Issue, Project card, checklist of deliverables.

Use the organization project space: `https://github.com/orgs/TheDataEngineX/projects`

______________________________________________________________________

### 2) Design

**Entry**: Approved plan.

**Activities**:

- Define interfaces, data contracts, and configuration changes.
- Update architecture or runbook docs if required.

**Exit**: Design reviewed and aligned.

**Artifacts**: Updated documentation, diagrams if applicable.

______________________________________________________________________

### 3) Implement

**Entry**: Approved design.

**Activities**:

- Develop in a short-lived feature branch.
- Add or update tests.
- Update docs if behavior or interfaces change.

**Exit**: Code complete and locally validated.

**Artifacts**: Code changes, tests, documentation updates.

______________________________________________________________________

### 4) Verify (CI)

**Entry**: Pull request opened against `main`.

**Required Checks**:

- CI, security, and quality checks as defined in the CI/CD pipeline.

See [CI/CD Pipeline](ci-cd.md) for the authoritative list of checks.

**Exit**: All checks pass and at least one reviewer approves.

**Artifacts**: CI logs, security scan reports, test results.

______________________________________________________________________

### 5) Release (CD)

**Entry**:

- PR merged to `main`.

**Actions**:

- Build wheel + sdist once.
- Publish to PyPI via OIDC trusted publishing.

**Exit**: Package published on PyPI and GitHub Release created.

**Artifacts**: Wheel + sdist, CycloneDX SBOM, GitHub Release.

______________________________________________________________________

### 6) Operate

**Entry**: Package published to PyPI.

**Activities**:

- Monitor PyPI download metrics and issue tracker.
- Respond to bug reports and feature requests.
- Patch security vulnerabilities in dependencies.

**Exit**: No active incidents.

**Artifacts**: GitHub Issues, dependency audit reports.

## Development Workflow (Summary)

```mermaid
flowchart TD
    Start([New Feature/Fix]) --> Issue[Create GitHub Issue]
    Issue --> Branch["Create branch: my-feature"]
    Branch --> Code[Implement + Tests]
    Code --> Local["Local checks: lint, test, types"]
    Local -->|Failed| Code
    Local -->|Passed| PR["Open PR → main"]
    PR --> CI{CI Checks}
    CI -->|Failed| Fix[Fix Issues]
    Fix --> Code
    CI -->|Passed| Review{Code Review}
    Review -->|Changes Requested| Code
    Review -->|Approved| MergeMain["Merge to main"]
    MergeMain --> Prod[Auto-release if tagged]
    Prod --> Monitor[Monitor]
    Monitor --> End([✓ Complete])
```

**Steps**:

1. Create a feature branch from `main`.
1. Implement changes, run local checks (`uv run poe check-all`).
1. Open a PR to `main` and request review.
1. Merge after all required checks pass.
1. Push a `vX.Y.Z` tag to trigger a PyPI release.

See [Contributing Guide](contributing.md) for branch naming, local checks, and PR conventions.

______________________________________________________________________

## Related Documentation

**Development:**

- **[Contributing Guide](contributing.md)** - Contribution workflow
- **[CI/CD Pipeline](ci-cd.md)** - Automated build and deploy

**Operations:**

- **Observability Guide** (see [observability.md](observability.md)) - Metrics, logging, tracing

______________________________________________________________________

**[← Back to Documentation](index.md)**
