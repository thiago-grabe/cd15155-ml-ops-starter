{
  "family": "finbert-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "3072",
  "executionRoleArn": "${EXEC_ROLE_ARN}",
  "runtimePlatform": {
    "cpuArchitecture": "X86_64",
    "operatingSystemFamily": "LINUX"
  },
  "containerDefinitions": [
    {
      "name": "finbert-api",
      "image": "${ECR_REGISTRY}/${ECR_REPOSITORY}:latest",
      "essential": true,
      "portMappings": [
        { "containerPort": 8000, "protocol": "tcp" }
      ],
      "environment": [
        { "name": "MODEL_SOURCE", "value": "huggingface" },
        { "name": "HF_MODEL_ID", "value": "baptle/FinBERT_market_based" },
        { "name": "API_HOST", "value": "0.0.0.0" },
        { "name": "API_PORT", "value": "8000" }
      ],
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "python -c \"import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).status==200 else 1)\" || exit 1"
        ],
        "interval": 30,
        "timeout": 10,
        "retries": 5,
        "startPeriod": 180
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/finbert-api",
          "awslogs-region": "${AWS_REGION}",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
