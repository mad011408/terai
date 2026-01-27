#!/bin/bash
# AI Terminal Agent - Setup Script
# This script sets up the development environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
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

# Check Python version
check_python() {
    print_header "Checking Python Version"

    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

        if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python3"
        else
            print_error "Python 3.10+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.10+"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_header "Creating Virtual Environment"

    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists"
        read -p "Do you want to recreate it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf venv
        else
            return
        fi
    fi

    $PYTHON_CMD -m venv venv
    print_success "Virtual environment created"
}

# Activate virtual environment
activate_venv() {
    print_header "Activating Virtual Environment"

    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_success "Virtual environment activated"
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
        print_success "Virtual environment activated"
    else
        print_error "Could not find activation script"
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    print_header "Installing Dependencies"

    # Upgrade pip
    pip install --upgrade pip
    print_success "pip upgraded"

    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Requirements installed"
    else
        print_warning "requirements.txt not found"
    fi

    # Install development dependencies
    read -p "Install development dependencies? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install -e ".[dev]"
        print_success "Development dependencies installed"
    fi
}

# Setup environment file
setup_env() {
    print_header "Setting Up Environment"

    if [ -f ".env" ]; then
        print_warning ".env file already exists"
        read -p "Do you want to overwrite it? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return
        fi
    fi

    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_success ".env file created from .env.example"
        print_warning "Please edit .env and add your API keys"
    else
        print_warning ".env.example not found"
    fi
}

# Create necessary directories
create_directories() {
    print_header "Creating Directories"

    directories=("logs" "data" ".cache" "tmp")

    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Created $dir/"
        fi
    done
}

# Setup pre-commit hooks
setup_hooks() {
    print_header "Setting Up Git Hooks"

    if command -v pre-commit &> /dev/null; then
        pre-commit install
        print_success "Pre-commit hooks installed"
    else
        print_warning "pre-commit not found, skipping hooks setup"
    fi
}

# Verify installation
verify_installation() {
    print_header "Verifying Installation"

    # Try importing main modules
    python -c "from src.core.agent import Agent; print('Core module OK')" 2>/dev/null && print_success "Core module" || print_error "Core module"
    python -c "from src.agents.manager_agent import ManagerAgent; print('Agents module OK')" 2>/dev/null && print_success "Agents module" || print_error "Agents module"
    python -c "from src.models.model_manager import ModelManager; print('Models module OK')" 2>/dev/null && print_success "Models module" || print_error "Models module"
}

# Run tests
run_tests() {
    print_header "Running Tests"

    read -p "Run test suite? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pytest tests/ -v --tb=short || print_warning "Some tests failed"
    fi
}

# Main setup function
main() {
    print_header "AI Terminal Agent Setup"

    echo "This script will set up the development environment."
    echo ""

    check_python
    create_venv
    activate_venv
    install_dependencies
    setup_env
    create_directories
    setup_hooks
    verify_installation
    run_tests

    print_header "Setup Complete!"

    echo "To get started:"
    echo ""
    echo "  1. Activate the virtual environment:"
    echo "     source venv/bin/activate"
    echo ""
    echo "  2. Edit .env with your API keys:"
    echo "     nano .env"
    echo ""
    echo "  3. Run the agent:"
    echo "     python main.py"
    echo ""
    print_success "Happy coding!"
}

# Run main
main "$@"
