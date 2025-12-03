# AI Terminal Agent - Deployment Guide

This guide covers deploying the AI Terminal Agent in various environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Production Deployment](#production-deployment)
5. [Cloud Deployments](#cloud-deployments)
6. [Configuration](#configuration)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8+ GB |
| Disk | 10 GB | 50+ GB |
| Python | 3.10 | 3.11+ |

### Required Software

- Python 3.10+
- pip or poetry
- Git
- Docker (for containerized deployment)

### API Keys

You'll need API keys for the LLM providers you plan to use:
- TryBons AI API key
- OpenAI API key (optional)
- Anthropic API key (optional)

---

## Local Development

### 1. Clone Repository

```bash
git clone https://github.com/example/ai-terminal-agent.git
cd ai-terminal-agent
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Using conda
conda create -n ai-agent python=3.11
conda activate ai-agent
```

### 3. Install Dependencies

```bash
# Production dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 4. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your API keys
nano .env  # or use your preferred editor
```

### 5. Run Application

```bash
# Interactive mode
python main.py

# Single task
python main.py "Your task here"

# With debug logging
python main.py --debug

# Run tests
pytest
```

---

## Docker Deployment

### Quick Start

```bash
# Build image
docker build -t ai-terminal-agent .

# Run container
docker run -it --rm \
  -e AI_AGENT_API_KEY=your_key \
  ai-terminal-agent
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f ai-agent

# Stop services
docker-compose down
```

### Docker Compose with Monitoring

```bash
# Include monitoring stack
docker-compose --profile monitoring up -d

# Access Grafana
open http://localhost:3000
```

### Custom Docker Build

```dockerfile
# Custom Dockerfile
FROM ai-terminal-agent:latest

# Add custom configurations
COPY custom-config/ /app/config/

# Add custom prompts
COPY custom-prompts/ /app/prompts/
```

---

## Production Deployment

### Security Checklist

- [ ] API keys stored securely (secrets manager)
- [ ] HTTPS enabled
- [ ] Rate limiting configured
- [ ] Logging enabled
- [ ] Monitoring configured
- [ ] Backup strategy in place
- [ ] Input validation enabled
- [ ] PII filtering enabled

### Recommended Architecture

```
┌─────────────────┐
│   Load Balancer │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌───▼───┐
│Agent 1│ │Agent 2│  (Multiple instances)
└───┬───┘ └───┬───┘
    │         │
    └────┬────┘
         │
┌────────▼────────┐
│   Redis Cache   │
└────────┬────────┘
         │
┌────────▼────────┐
│   PostgreSQL    │
└─────────────────┘
```

### Systemd Service

```ini
# /etc/systemd/system/ai-agent.service
[Unit]
Description=AI Terminal Agent
After=network.target

[Service]
Type=simple
User=aiagent
WorkingDirectory=/opt/ai-terminal-agent
Environment=PATH=/opt/ai-terminal-agent/venv/bin
EnvironmentFile=/opt/ai-terminal-agent/.env
ExecStart=/opt/ai-terminal-agent/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable ai-agent
sudo systemctl start ai-agent
sudo systemctl status ai-agent
```

### Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/ai-agent
upstream ai_agent {
    server 127.0.0.1:8000;
}

server {
    listen 443 ssl http2;
    server_name agent.example.com;

    ssl_certificate /etc/letsencrypt/live/agent.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/agent.example.com/privkey.pem;

    location / {
        proxy_pass http://ai_agent;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Cloud Deployments

### AWS

#### ECS Deployment

```yaml
# task-definition.json
{
  "family": "ai-terminal-agent",
  "containerDefinitions": [
    {
      "name": "ai-agent",
      "image": "your-ecr-repo/ai-terminal-agent:latest",
      "cpu": 512,
      "memory": 1024,
      "essential": true,
      "secrets": [
        {
          "name": "AI_AGENT_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:..."
        }
      ]
    }
  ]
}
```

#### Lambda Deployment

```python
# lambda_handler.py
import json
from main import run_single_task

def handler(event, context):
    task = event.get('task', '')
    result = run_single_task(task)
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

### Google Cloud

#### Cloud Run

```bash
# Deploy to Cloud Run
gcloud run deploy ai-terminal-agent \
  --image gcr.io/your-project/ai-terminal-agent \
  --platform managed \
  --region us-central1 \
  --set-env-vars AI_AGENT_API_KEY=your_key
```

### Azure

#### Container Instances

```bash
# Deploy to Azure Container Instances
az container create \
  --resource-group myResourceGroup \
  --name ai-terminal-agent \
  --image your-registry.azurecr.io/ai-terminal-agent:latest \
  --cpu 2 \
  --memory 4 \
  --environment-variables AI_AGENT_API_KEY=your_key
```

### Kubernetes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-terminal-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-agent
  template:
    metadata:
      labels:
        app: ai-agent
    spec:
      containers:
      - name: ai-agent
        image: your-registry/ai-terminal-agent:latest
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        env:
        - name: AI_AGENT_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-agent-secrets
              key: api-key
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AI_AGENT_API_KEY` | API key for LLM service | Required |
| `AI_AGENT_API_HOST` | API host URL | `https://go.trybons.ai` |
| `AI_AGENT_DEFAULT_MODEL` | Default model | `anthropic/claude-sonnet-4` |
| `AI_AGENT_LOG_LEVEL` | Log level | `INFO` |
| `AI_AGENT_MAX_ITERATIONS` | Max agent iterations | `10` |
| `AI_AGENT_TIMEOUT` | Request timeout (seconds) | `300` |

### Configuration Files

```yaml
# config/agents.yaml
agents:
  manager:
    model: anthropic/claude-sonnet-4
    max_iterations: 10
    temperature: 0.7
```

---

## Monitoring

### Prometheus Metrics

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'ai-agent'
    static_configs:
      - targets: ['ai-agent:8000']
```

### Grafana Dashboards

Import the pre-built dashboards from `monitoring/grafana/dashboards/`.

### Health Checks

```bash
# HTTP health check
curl http://localhost:8000/health

# Docker health check (built-in)
docker inspect --format='{{.State.Health.Status}}' ai-terminal-agent
```

### Logging

```python
# Structured logging output
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "message": "Task completed",
  "agent": "code",
  "duration_ms": 1234,
  "success": true
}
```

---

## Troubleshooting

### Common Issues

#### API Connection Failed

```bash
# Check connectivity
curl -v https://go.trybons.ai/v1/chat/completions

# Verify API key
echo $AI_AGENT_API_KEY
```

#### Out of Memory

```bash
# Increase memory limit
docker run -m 4g ai-terminal-agent

# Check memory usage
docker stats ai-terminal-agent
```

#### Slow Response Times

1. Check network latency to API
2. Review cache hit rates
3. Monitor model response times
4. Check for rate limiting

#### Permission Denied

```bash
# Fix file permissions
chmod -R 755 /app
chown -R agent:agent /app
```

### Debug Mode

```bash
# Enable debug logging
python main.py --debug

# Verbose Docker logs
docker-compose logs -f --tail=100 ai-agent
```

### Support

- GitHub Issues: Report bugs and request features
- Documentation: Check docs/ directory
- Logs: Review application logs for errors

---

## Backup and Recovery

### Data Backup

```bash
# Backup volumes
docker run --rm \
  -v ai-terminal-agent_agent-data:/data \
  -v $(pwd)/backup:/backup \
  alpine tar czf /backup/data.tar.gz /data

# Backup database
docker exec ai-agent-postgres pg_dump -U agent ai_agent > backup.sql
```

### Recovery

```bash
# Restore data
docker run --rm \
  -v ai-terminal-agent_agent-data:/data \
  -v $(pwd)/backup:/backup \
  alpine tar xzf /backup/data.tar.gz -C /

# Restore database
cat backup.sql | docker exec -i ai-agent-postgres psql -U agent ai_agent
```

---

## Scaling Guidelines

### Horizontal Scaling

- Use Redis for shared state
- Configure load balancer
- Ensure stateless agent design

### Vertical Scaling

- Increase container resources
- Optimize batch sizes
- Tune concurrency settings

### Auto-scaling

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-terminal-agent
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```
