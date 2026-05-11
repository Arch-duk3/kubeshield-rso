#!/bin/bash
echo "Starting KRSI Optimization Framework..."

echo "Building Go Simulator..."
cd simulator
go build -o simulator_app main.go
./simulator_app &
SIM_PID=$!
cd ..

echo "Setting up Python Controller..."
cd controller
echo "Running RL Agent..."
python3 main.py

kill $SIM_PID
echo "Done."
