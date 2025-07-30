#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🧪 MediMaven Test Suite${NC}"
echo "========================"

# Parse arguments
RUN_E2E=false
RUN_COVERAGE=false
WATCH_MODE=false

for arg in "$@"; do
  case $arg in
    --e2e) RUN_E2E=true ;;
    --coverage) RUN_COVERAGE=true ;;
    --watch) WATCH_MODE=true ;;
    --help)
      echo "Usage: ./test.sh [options]"
      echo "Options:"
      echo "  --e2e       Run E2E tests"
      echo "  --coverage  Generate coverage reports"
      echo "  --watch     Run tests in watch mode"
      exit 0
      ;;
  esac
done

# Check if we're in Colab
if [[ -d "/content/drive" ]]; then
  echo -e "${BLUE}Detected Google Colab environment${NC}"
  echo "Running Colab-compatible tests..."
  
  cd /content/drive/Othercomputers/kyei_mac_m3_pro/MediMaven/backend
  python tests/run_colab_tests.py
  exit $?
fi

# Frontend Tests
echo -e "\n${BLUE}📱 Frontend Tests${NC}"
cd frontend

if [ "$WATCH_MODE" = true ]; then
  npm test
elif [ "$RUN_COVERAGE" = true ]; then
  npm test -- --run --coverage
else
  npm test -- --run
fi

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✓ Frontend tests passed${NC}"
else
  echo -e "${RED}✗ Frontend tests failed${NC}"
  exit 1
fi

# Backend Tests
echo -e "\n${BLUE}🔧 Backend Tests${NC}"
cd ../backend

# Set test environment
export DATABASE_URL="sqlite+aiosqlite:///test.db"
export ENABLE_MONITORING="false"
export JWT_SECRET="test-secret-key"

if [ "$RUN_COVERAGE" = true ]; then
  pytest -v --cov --cov-report=term-missing
else
  pytest -v
fi

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✓ Backend tests passed${NC}"
else
  echo -e "${RED}✗ Backend tests failed${NC}"
  exit 1
fi

# E2E Tests (optional)
if [ "$RUN_E2E" = true ]; then
  echo -e "\n${BLUE}🌐 E2E Tests${NC}"
  
  # Start backend
  cd ../backend
  uvicorn app.main:app --host 0.0.0.0 --port 8000 &
  BACKEND_PID=$!
  
  # Start frontend
  cd ../frontend
  npm run dev &
  FRONTEND_PID=$!
  
  # Wait for services
  sleep 10
  
  # Run Cypress
  npx cypress run
  E2E_RESULT=$?
  
  # Cleanup
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
  
  if [ $E2E_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ E2E tests passed${NC}"
  else
    echo -e "${RED}✗ E2E tests failed${NC}"
    exit 1
  fi
fi

echo -e "\n${GREEN}✅ All tests passed!${NC}"

# Coverage report location
if [ "$RUN_COVERAGE" = true ]; then
  echo -e "\n${BLUE}📊 Coverage Reports:${NC}"
  echo "  Frontend: frontend/coverage/index.html"
  echo "  Backend: backend/htmlcov/index.html"
fi
