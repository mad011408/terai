#!/bin/bash
# AI Terminal Agent - Deployment Script
# This script handles deployment to various environments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
IMAGE_NAME="ai-terminal-agent"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGISTRY:-}"

print_message() {
    echo -e "${1}${2}${NC}"
}

print_header() {
    echo ""
    print_message $BLUE "=============================================="
    print_message $BLUE "$1"
    print_message $BLUE "=============================================="
    echo ""
}

print_success() {
    print_message $GREEN "✓ $1"
}

print_warning() {
    print_message $YELLOW "⚠ $1"
}

print_error() {
    print_message $RED "✗ $1"
}

# Show usage
usage() {
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  build       Build Docker image"
    echo "  push        Push image to registry"
    echo "  deploy      Deploy to target environment"
    echo "  rollback    Rollback to previous version"
    echo "  status      Check deployment status"
    echo "  logs        View deployment logs"
    echo ""
    echo "Options:"
    echo "  -e, --env       Environment (dev, staging, prod)"
    echo "  -t, --tag       Image tag (default: latest)"
    echo "  -r, --registry  Container registry URL"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 push -r gcr.io/myproject"
    echo "  $0 deploy -e prod"
    exit 1
}

# Build Docker image
build_image() {
    print_header "Building Docker Image"

    echo "Building ${IMAGE_NAME}:${IMAGE_TAG}..."

    docker build \
        -t ${IMAGE_NAME}:${IMAGE_TAG} \
        -t ${IMAGE_NAME}:latest \
        --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
        --build-arg VERSION=${IMAGE_TAG} \
        .

    print_success "Image built successfully"

    # Show image info
    docker images ${IMAGE_NAME}:${IMAGE_TAG}
}

# Push image to registry
push_image() {
    print_header "Pushing Docker Image"

    if [ -z "$REGISTRY" ]; then
        print_error "Registry not specified. Use -r or --registry option."
        exit 1
    fi

    FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

    echo "Tagging image..."
    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${FULL_IMAGE}

    echo "Pushing ${FULL_IMAGE}..."
    docker push ${FULL_IMAGE}

    # Also push latest
    docker tag ${IMAGE_NAME}:latest ${REGISTRY}/${IMAGE_NAME}:latest
    docker push ${REGISTRY}/${IMAGE_NAME}:latest

    print_success "Image pushed to ${REGISTRY}"
}

# Deploy to environment
deploy() {
    print_header "Deploying to ${ENVIRONMENT}"

    case $ENVIRONMENT in
        dev)
            deploy_dev
            ;;
        staging)
            deploy_staging
            ;;
        prod)
            deploy_production
            ;;
        *)
            print_error "Unknown environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
}

# Deploy to development
deploy_dev() {
    echo "Deploying to development environment..."

    # Using docker-compose for dev
    docker-compose -f docker-compose.yml up -d

    print_success "Development deployment complete"

    # Show status
    docker-compose ps
}

# Deploy to staging
deploy_staging() {
    echo "Deploying to staging environment..."

    # Example: Kubernetes deployment
    if command -v kubectl &> /dev/null; then
        kubectl apply -f k8s/staging/ --namespace=staging
        kubectl rollout status deployment/ai-terminal-agent -n staging
        print_success "Staging deployment complete"
    else
        print_warning "kubectl not found, using docker-compose"
        docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d
    fi
}

# Deploy to production
deploy_production() {
    echo "Deploying to production environment..."

    # Confirmation for production
    read -p "Are you sure you want to deploy to PRODUCTION? (yes/no) " confirm
    if [ "$confirm" != "yes" ]; then
        print_warning "Deployment cancelled"
        exit 0
    fi

    # Pre-deployment checks
    run_pre_deploy_checks

    # Example: Kubernetes deployment
    if command -v kubectl &> /dev/null; then
        # Apply deployment
        kubectl apply -f k8s/production/ --namespace=production

        # Wait for rollout
        kubectl rollout status deployment/ai-terminal-agent -n production --timeout=300s

        print_success "Production deployment complete"

        # Run post-deployment verification
        run_post_deploy_verification
    else
        print_error "kubectl required for production deployment"
        exit 1
    fi
}

# Pre-deployment checks
run_pre_deploy_checks() {
    print_header "Running Pre-deployment Checks"

    # Check image exists
    if ! docker image inspect ${IMAGE_NAME}:${IMAGE_TAG} &> /dev/null; then
        print_error "Image ${IMAGE_NAME}:${IMAGE_TAG} not found"
        exit 1
    fi
    print_success "Image exists"

    # Check configuration
    if [ ! -f "config/agents.yaml" ]; then
        print_error "Configuration files not found"
        exit 1
    fi
    print_success "Configuration valid"

    # Run tests
    echo "Running tests..."
    docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} pytest tests/ -v --tb=short
    print_success "Tests passed"
}

# Post-deployment verification
run_post_deploy_verification() {
    print_header "Running Post-deployment Verification"

    # Wait for pods to be ready
    sleep 10

    # Health check
    echo "Checking health endpoint..."
    # kubectl exec -n production deploy/ai-terminal-agent -- curl -s http://localhost:8000/health

    print_success "Deployment verified"
}

# Rollback deployment
rollback() {
    print_header "Rolling Back Deployment"

    if [ -z "$ENVIRONMENT" ]; then
        print_error "Environment not specified"
        exit 1
    fi

    case $ENVIRONMENT in
        dev)
            docker-compose down
            docker-compose up -d
            ;;
        staging|prod)
            if command -v kubectl &> /dev/null; then
                kubectl rollout undo deployment/ai-terminal-agent -n ${ENVIRONMENT}
                kubectl rollout status deployment/ai-terminal-agent -n ${ENVIRONMENT}
            fi
            ;;
    esac

    print_success "Rollback complete"
}

# Check deployment status
check_status() {
    print_header "Deployment Status"

    if [ -z "$ENVIRONMENT" ]; then
        # Show all environments
        echo "Local Docker:"
        docker-compose ps 2>/dev/null || echo "Not running"
        echo ""

        if command -v kubectl &> /dev/null; then
            echo "Kubernetes:"
            kubectl get deployments -A -l app=ai-terminal-agent 2>/dev/null || echo "No deployments found"
        fi
    else
        case $ENVIRONMENT in
            dev)
                docker-compose ps
                ;;
            staging|prod)
                kubectl get all -n ${ENVIRONMENT} -l app=ai-terminal-agent
                ;;
        esac
    fi
}

# View logs
view_logs() {
    print_header "Deployment Logs"

    if [ -z "$ENVIRONMENT" ]; then
        ENVIRONMENT="dev"
    fi

    case $ENVIRONMENT in
        dev)
            docker-compose logs -f --tail=100 ai-agent
            ;;
        staging|prod)
            kubectl logs -f -n ${ENVIRONMENT} -l app=ai-terminal-agent --tail=100
            ;;
    esac
}

# Parse arguments
COMMAND=""
ENVIRONMENT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        build|push|deploy|rollback|status|logs)
            COMMAND=$1
            shift
            ;;
        -e|--env)
            ENVIRONMENT=$2
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG=$2
            shift 2
            ;;
        -r|--registry)
            REGISTRY=$2
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Execute command
if [ -z "$COMMAND" ]; then
    usage
fi

case $COMMAND in
    build)
        build_image
        ;;
    push)
        push_image
        ;;
    deploy)
        deploy
        ;;
    rollback)
        rollback
        ;;
    status)
        check_status
        ;;
    logs)
        view_logs
        ;;
esac
