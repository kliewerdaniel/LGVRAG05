#!/bin/bash

# Enhanced RAG System Deployment Script
# This script automates the deployment of the RAG system with proper
# environment setup, dependency management, and health checks.

set -e  # Exit on any error

# Configuration
PROJECT_NAME="enhanced-rag-system"
DEPLOY_ENV="${DEPLOY_ENV:-production}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-60}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."

    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    log_info "Python version: $PYTHON_VERSION"

    # Check available disk space
    DISK_SPACE=$(df . | tail -1 | awk '{print $4}')
    if [ "$DISK_SPACE" -lt 1000000 ]; then  # Less than 1GB
        log_error "Insufficient disk space. Need at least 1GB free."
        exit 1
    fi
    log_info "Available disk space: $((DISK_SPACE/1000000))GB"

    # Check available memory
    if command -v free &> /dev/null; then
        MEM_GB=$(free -g | awk 'NR==2{printf "%.0f", $2}')
        if [ "$MEM_GB" -lt 4 ]; then
            log_warning "Low memory detected: ${MEM_GB}GB. Recommended: 8GB+"
        else
            log_info "Available memory: ${MEM_GB}GB"
        fi
    fi

    log_success "Pre-deployment checks completed"
}

# Setup virtual environment
setup_environment() {
    log_info "Setting up Python environment..."

    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv .venv
    fi

    # Activate virtual environment
    source .venv/bin/activate

    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip

    log_success "Environment setup completed"
}

# Install dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."

    source .venv/bin/activate

    # Install core dependencies
    log_info "Installing core dependencies..."
    pip install --break-system-packages -r requirements.txt

    # Install development dependencies for production
    if [ "$DEPLOY_ENV" = "production" ]; then
        log_info "Installing production dependencies..."
        pip install --break-system-packages gunicorn uvicorn[standard]
    fi

    log_success "Dependencies installed successfully"
}

# Setup configuration
setup_configuration() {
    log_info "Setting up configuration..."

    source .venv/bin/activate

    # Create necessary directories
    mkdir -p data/chromadb
    mkdir -p data/logs
    mkdir -p evaluation/results
    mkdir -p temp_uploads

    # Set proper permissions
    chmod 755 data/chromadb
    chmod 755 data/logs

    # Generate default configuration if it doesn't exist
    if [ ! -f "config.yaml" ]; then
        log_info "Generating default configuration..."
        python3 -c "from config import create_default_config; create_default_config()"
    fi

    # Validate configuration
    log_info "Validating configuration..."
    python3 -c "from config import get_config; config = get_config(); print('Configuration loaded successfully')"

    log_success "Configuration setup completed"
}

# Run database migrations/setup
setup_database() {
    log_info "Setting up database..."

    source .venv/bin/activate

    # Initialize ChromaDB
    log_info "Initializing ChromaDB..."
    python3 -c "
from db.helix_interface import ChromaDBInterface
import logging
logging.basicConfig(level=logging.INFO)
db = ChromaDBInterface()
print('ChromaDB initialized successfully')
"

    log_success "Database setup completed"
}

# Run tests
run_tests() {
    log_info "Running tests..."

    source .venv/bin/activate

    # Run basic functionality tests
    log_info "Running functionality tests..."
    python3 -c "
from rag.bm25_search import BM25Search
from rag.cross_encoder_rerank import CrossEncoderReranker
from config import get_config
print('✓ All imports successful')

# Test BM25
bm25 = BM25Search([{'id': 'test', 'text': 'test document'}])
print('✓ BM25 initialization successful')

# Test cross-encoder
reranker = CrossEncoderReranker()
print('✓ Cross-encoder initialization successful')

# Test configuration
config = get_config()
print('✓ Configuration loading successful')

print('All tests passed!')
"

    log_success "Tests completed successfully"
}

# Start services
start_services() {
    log_info "Starting RAG system services..."

    source .venv/bin/activate

    # Start Ollama if not running (optional)
    if command -v ollama &> /dev/null; then
        if ! pgrep -x "ollama" > /dev/null; then
            log_info "Starting Ollama service..."
            ollama serve &
            OLLAMA_PID=$!
            sleep 5

            # Wait for Ollama to be ready
            timeout=30
            while [ $timeout -gt 0 ]; do
                if curl -s http://localhost:11434/api/tags > /dev/null; then
                    log_success "Ollama service is ready"
                    break
                fi
                sleep 1
                timeout=$((timeout-1))
            done

            if [ $timeout -eq 0 ]; then
                log_warning "Ollama service may not be fully ready"
            fi
        else
            log_info "Ollama service is already running"
        fi
    fi

    # Start the API server
    log_info "Starting FastAPI server..."
    nohup python3 -c "
import uvicorn
import sys
sys.path.append('.')
from config import get_config
config = get_config()
uvicorn.run('api.main:app', host=config.api.host, port=config.api.port, reload=config.api.reload)
" > api.log 2>&1 &
    API_PID=$!
    sleep 3

    log_success "Services started"
}

# Health check
health_check() {
    log_info "Performing health checks..."

    source .venv/bin/activate

    # Check API health
    log_info "Checking API health..."
    HEALTH_CHECK_URL="http://localhost:8000/health"

    for i in $(seq 1 $HEALTH_CHECK_TIMEOUT); do
        if curl -s -f $HEALTH_CHECK_URL > /dev/null 2>&1; then
            log_success "API health check passed"
            break
        fi

        if [ $i -eq $HEALTH_CHECK_TIMEOUT ]; then
            log_error "API health check failed after $HEALTH_CHECK_TIMEOUT seconds"
            exit 1
        fi

        sleep 1
    done

    # Check system metrics
    log_info "Checking system metrics..."
    python3 -c "
from monitoring.observability import get_system_health
health = get_system_health()
print(f'System status: {health[\"status\"]}')
if health['issues']:
    print(f'Issues: {health[\"issues\"]}')
else:
    print('No issues detected')
"

    log_success "All health checks passed"
}

# Main deployment function
main() {
    log_info "Starting deployment of $PROJECT_NAME ($DEPLOY_ENV)..."

    # Run deployment steps
    pre_deployment_checks
    setup_environment
    install_dependencies
    setup_configuration
    setup_database
    run_tests
    start_services
    health_check

    log_success "Deployment completed successfully!"
    log_info "RAG system is ready and accessible at http://localhost:8000"
    log_info "API documentation available at http://localhost:8000/docs"

    # Print next steps
    echo ""
    log_info "Next steps:"
    echo "1. Access the web interface at http://localhost:8000"
    echo "2. Upload documents through the API or web interface"
    echo "3. Start asking questions about your documents"
    echo "4. Monitor system health at http://localhost:8000/api/health"
    echo ""
    log_info "To stop the system, run: pkill -f 'python.*api'"
    log_info "Monitor system health at http://localhost:8000/health"
}

# Handle script arguments
case "${1:-}" in
    "health")
        source .venv/bin/activate 2>/dev/null || true
        health_check
        ;;
    "test")
        source .venv/bin/activate 2>/dev/null || true
        run_tests
        ;;
    "stop")
        log_info "Stopping RAG system..."
        pkill -f "python.*api" || true
        pkill -f "ollama" || true
        log_success "System stopped"
        ;;
    *)
        main
        ;;
esac
