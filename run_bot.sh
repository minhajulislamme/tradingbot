#!/bin/bash

# Run the LAYER Trading Bot with small-account flag
# This script activates the virtual environment and runs the trading bot

# Get the absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Ensure virtual environment exists
if [ ! -d "${SCRIPT_DIR}/venv" ]; then
    echo -e "${RED}Virtual environment not found. Run setup.sh first.${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source "${SCRIPT_DIR}/venv/bin/activate"

# Check if the bot is already running
if pgrep -f "python3 ${SCRIPT_DIR}/main.py" > /dev/null; then
    echo -e "${YELLOW}LAYER Trading bot is already running. Use check_status.sh to see its status.${NC}"
    exit 0
fi

# Create logs directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/logs"

# Start the bot with small-account flag
echo -e "${GREEN}Starting LAYER trading bot with small-account settings...${NC}"
echo -e "${YELLOW}Bot will run in the background. Check logs for status.${NC}"

# Run the bot with nohup to keep it running after SSH session closes
nohup python3 "${SCRIPT_DIR}/main.py" > "${SCRIPT_DIR}/logs/bot_console.log" 2>&1 &

# Save the PID to a file for later use
echo $! > "${SCRIPT_DIR}/bot.pid"

echo -e "${GREEN}LAYER Bot started with PID $(cat ${SCRIPT_DIR}/bot.pid)${NC}"
echo -e "${GREEN}Use check_status.sh to see the bot's status${NC}"
echo -e "${GREEN}Use stop_bot.sh to stop the bot${NC}"