# TechCorp API Reference

## Authentication

All API requests require authentication via Bearer token. Include your API key in the Authorization header:

```
Authorization: Bearer tc_live_abc123xyz
```

API keys can be generated from the TechCorp Dashboard under Settings > API Keys. Each key has configurable permissions and rate limits.

## Rate Limits

| Plan | Requests/minute | Requests/day |
|------|----------------|--------------|
| Free | 60 | 1,000 |
| Pro | 600 | 50,000 |
| Enterprise | 6,000 | Unlimited |

Rate limit headers are included in every response:
- `X-RateLimit-Limit`: Maximum requests per minute
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Unix timestamp when the window resets

## Endpoints

### GET /api/v2/instances

List all cloud instances managed by TechCorp.

**Parameters:**
- `region` (optional): Filter by region (e.g., `us-east-1`, `eu-west-1`)
- `status` (optional): Filter by status (`running`, `stopped`, `terminated`)
- `page` (optional, default: 1): Page number for pagination
- `per_page` (optional, default: 50, max: 200): Results per page

**Response:**
```json
{
  "data": [
    {
      "id": "inst_abc123",
      "name": "web-server-prod-01",
      "type": "c5.2xlarge",
      "region": "us-east-1",
      "status": "running",
      "monthly_cost": 245.50,
      "cpu_utilization": 0.42,
      "created_at": "2026-01-15T08:30:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 50,
    "total": 234,
    "total_pages": 5
  }
}
```

### POST /api/v2/instances/{id}/optimize

Run cost optimization analysis on a specific instance.

**Response:**
```json
{
  "instance_id": "inst_abc123",
  "current_cost": 245.50,
  "recommended_type": "c5.xlarge",
  "projected_cost": 122.75,
  "savings_percent": 50.0,
  "confidence": 0.92,
  "reasoning": "CPU utilization averaging 42% over 30 days suggests overprovisioning."
}
```

### GET /api/v2/costs/summary

Get a cost summary for your organization.

**Parameters:**
- `start_date` (required): Start of date range (YYYY-MM-DD)
- `end_date` (required): End of date range (YYYY-MM-DD)
- `group_by` (optional): Group results by `service`, `region`, `team`, or `tag`

### POST /api/v2/alerts

Create a cost or performance alert.

**Request body:**
```json
{
  "name": "High CPU Alert",
  "type": "performance",
  "condition": {
    "metric": "cpu_utilization",
    "operator": "greater_than",
    "threshold": 0.85,
    "duration_minutes": 15
  },
  "notification": {
    "channels": ["email", "slack"],
    "recipients": ["ops-team@techcorp.com"]
  }
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request — Invalid parameters |
| 401 | Unauthorized — Invalid or missing API key |
| 403 | Forbidden — Insufficient permissions |
| 404 | Not Found — Resource does not exist |
| 429 | Too Many Requests — Rate limit exceeded |
| 500 | Internal Server Error — Contact support |

## SDKs

Official SDKs are available for:
- Python: `pip install techcorp-sdk`
- JavaScript/TypeScript: `npm install @techcorp/sdk`
- Go: `go get github.com/techcorp/sdk-go`
