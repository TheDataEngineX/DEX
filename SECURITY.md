# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest (0.7.x) | ✅ |
| Previous minor release | ✅ (security fixes only) |
| Older versions | ❌ |

## Reporting a Vulnerability

**Do NOT open a public issue for security vulnerabilities.**

Instead, please report them via one of these channels:

1. **Email**: security@thedataenginex.dev
1. **GitHub Security Advisories**: Use the "Report a vulnerability" button on the Security tab

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested fix (if any)

### Response Timeline

| Stage | Timeline |
|-------|----------|
| Acknowledgment | Within 48 hours |
| Initial assessment | Within 5 business days |
| Fix development | Within 30 days (critical), 90 days (non-critical) |
| Public disclosure | After fix is released |

## Disclosure Policy

We follow [coordinated disclosure](https://en.wikipedia.org/wiki/Coordinated_vulnerability_disclosure).
We will credit reporters in the security advisory unless they prefer to remain anonymous.

## Security Practices

DataEngineX follows these security practices:

- **No hardcoded secrets** — all credentials via environment variables
- **Parameterized queries** — never SQL concatenation
- **Input validation** — Pydantic models at API boundaries
- **Dependency auditing** — automated via `uv run poe security`
- **Pickle safety** — SafeUnpickler with HMAC verification for model loading
- **Container security** — non-root users, minimal base images
- **HTTPS only** — all production traffic encrypted
- **Least privilege** — minimal permissions for service accounts

## Security-Related Dependencies

*This table is illustrative, not exhaustive.*

| Dependency | Purpose | Security Note |
|------------|---------|---------------|
| pydantic | Config validation | Validates all inputs |
| python-dotenv | Env var loading | Never commit .env files |
| httpx | HTTP client | Timeout configured |
| structlog | Logging | No PII in logs by default |

## Auditing

Run security audits locally:

```bash
uv run poe security  # pip-audit for vulnerabilities
```

CI runs `pip-audit` and dependency scanning on every PR.
