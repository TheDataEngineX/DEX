# Security Scanning

DataEngineX uses a multi-layered security scanning pipeline across CI/CD workflows.

## Tools

| Tool | Purpose | Workflow |
|------|---------|----------|
| **Trivy** | Repository misconfiguration & secret scanning | `security.yml` |
| **CodeQL** | Static analysis (Python + GitHub Actions) | `security.yml` |
| **pip-audit** | Python dependency vulnerability audit | `poe security` |
| **CycloneDX** | SBOM generation (attached to releases) | `release.yml` |
| **Renovate** | Automated dependency updates | `renovate.json` |

## How It Works

### On Every Push / PR (security.yml)

1. **Trivy repo scan** — scans the filesystem for misconfigurations and leaked secrets (SARIF uploaded to GitHub Security tab).
1. **Trivy misconfig gate** — fails the build if HIGH or CRITICAL misconfigurations are found.
1. **CodeQL** — static analysis for Python code and GitHub Actions workflows.

### On Release (release.yml)

1. **CycloneDX SBOM** — generates a Software Bill of Materials in CycloneDX JSON format.
1. **SBOM upload** — attaches the SBOM as a release asset on GitHub Releases.

### Local / CI (poe tasks)

```bash
uv run poe security    # Run pip-audit against installed dependencies
```

## Reviewing Results

- **GitHub Security tab** — Trivy SARIF results and CodeQL alerts appear under *Security → Code scanning alerts*.
- **GitHub Releases** — each release includes an `sbom-*.json` CycloneDX artifact.
- **CI logs** — Trivy table output is visible in workflow run logs.
- **Dependabot** — automated PRs for vulnerable or outdated dependencies (pip + GitHub Actions, weekly).

## Acting on Findings

1. **CRITICAL / HIGH vulnerabilities** — fix immediately. Trivy and pip-audit will block the build.
1. **MEDIUM / LOW vulnerabilities** — triage in the next sprint. Document accepted risks.
1. **SBOM distribution** — share the CycloneDX JSON with downstream consumers for supply-chain transparency.
1. **Dependabot PRs** — review and merge promptly; they appear as standard pull requests.

## PrivacyGuard (Runtime PII Protection)

In addition to CI scanning, DataEngineX includes **PrivacyGuard** — a runtime security primitive that intercepts every outbound LLM call:

- **PII detection** — regex + NER models identify sensitive data (emails, SSNs, credit cards, etc.)
- **Masking strategies** — mask, redact, hash, or allow per entity type
- **Audit log** — every outbound call is recorded with PII types and actions taken
- **Zero-config** — enabled by default in `dex.yaml` under `secops.guard`

See [Architecture → Security](architecture.md#security--privacyguard) for details.
