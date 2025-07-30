#!/bin/bash

# Script to start both frontend and mock backend for E2E testing

echo "Starting E2E test servers..."

# Build frontend first
echo "Building frontend..."
npm run build

# Start mock backend in background
echo "Starting mock backend on port 8000..."
python mock_backend.py &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 3

# Start frontend preview server in background
echo "Starting frontend on port 5173..."
npm run preview -- --host 0.0.0.0 --port 5173 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

# Wait for servers to be ready
echo "Waiting for servers to be ready..."
sleep 5

# Check if servers are running
echo "Checking server health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend is running"
else
    echo "❌ Backend is not responding"
fi

if curl -f http://localhost:5173 > /dev/null 2>&1; then
    echo "✅ Frontend is running"
else
    echo "❌ Frontend is not responding"
fi

echo ""
echo "Servers are ready! You can now run:"
echo "  npm run cypress:run    # Run E2E tests in headless mode"
echo "  npm run cypress:open   # Open Cypress UI"
echo ""
echo "To stop the servers, run:"
echo "  kill $BACKEND_PID $FRONTEND_PID"
echo ""
echo "Press Ctrl+C to stop both servers and exit"

# Keep script running and handle cleanup
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "Servers stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for user to stop
wait
