# TechCorp Security Guide

## Data Protection

TechCorp implements multiple layers of security to protect your data:

### Encryption

- **At rest:** All data is encrypted using AES-256-GCM. Encryption keys are managed by AWS KMS with automatic key rotation every 90 days.
- **In transit:** All connections use TLS 1.3. We enforce HSTS with a minimum age of one year. Certificate pinning is available for mobile applications.
- **Application-level:** Sensitive fields (API keys, passwords, PII) are additionally encrypted with envelope encryption before storage.

### Access Control

TechCorp uses a Role-Based Access Control (RBAC) model with the following default roles:

| Role | Permissions |
|------|------------|
| Viewer | Read-only access to dashboards and reports |
| Operator | Viewer + ability to manage instances and alerts |
| Admin | Operator + user management and billing |
| Owner | Full access including API key management and account deletion |

Custom roles can be created with granular permissions. All role changes are logged in the audit trail.

### Audit Logging

Every action in the TechCorp Platform is logged with:
- Timestamp (UTC)
- User ID and IP address
- Action performed
- Resource affected
- Result (success/failure)

Audit logs are retained for 2 years and can be exported via the API or downloaded from the Dashboard. Logs are immutable and tamper-evident using cryptographic chaining.

## Compliance

TechCorp maintains the following certifications and compliance standards:

- **SOC 2 Type II** — Audited annually by Deloitte
- **ISO 27001** — Information security management
- **GDPR** — Full compliance for EU data subjects
- **HIPAA** — Available for healthcare customers (requires BAA)
- **PCI DSS Level 1** — For customers processing payment data

## Incident Response

Our incident response process follows the NIST framework:

1. **Detection:** Automated monitoring with <5 minute detection time for critical issues
2. **Containment:** Affected systems are isolated within 15 minutes
3. **Eradication:** Root cause analysis and remediation within 4 hours
4. **Recovery:** Service restoration with full data integrity verification
5. **Lessons Learned:** Post-incident review within 48 hours, published to affected customers

### Security Contact

Report security vulnerabilities to security@techcorp.com. We operate a bug bounty program with rewards up to $50,000 for critical vulnerabilities. We commit to acknowledging reports within 24 hours and providing a resolution timeline within 72 hours.

## Network Security

- DDoS protection via Cloudflare Enterprise
- Web Application Firewall (WAF) with custom rule sets
- Network segmentation between production, staging, and development environments
- VPN access required for all internal systems
- Regular penetration testing by NCC Group (quarterly)
