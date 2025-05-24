#!/bin/bash

# Stop the Binance Trading Bot that was started with run_bot.sh
# This script stops the bot process properly

# Get the absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if the bot PID file exists fun
if [ ! -f "${SCRIPT_DIR}/bot.pid" ]; then
    echo -e "${YELLOW}Bot PID file not found. The bot might not be running.${NC}"
    
    # Check if we can find the bot process anyway
    BOT_PID=$(pgrep -f "python3 ${SCRIPT_DIR}/main.py")
    if [ -z "$BOT_PID" ]; then
        echo -e "${RED}No running bot process found.${NC}"
        exit 1
    else
        echo -e "${GREEN}Found bot process with PID ${BOT_PID}${NC}"
    fi
else
    BOT_PID=$(cat "${SCRIPT_DIR}/bot.pid")
    echo -e "${GREEN}Found bot PID file: ${BOT_PID}${NC}"
fi

# Try to stop the bot gracefully first
echo -e "${YELLOW}Sending SIGTERM to bot process...${NC}"
kill -TERM $BOT_PID 2>/dev/null

# Wait up to 10 seconds for the process to exit gracefully
for i in {1..10}; do
    if ! ps -p $BOT_PID > /dev/null; then
        echo -e "${GREEN}Bot has stopped gracefully.${NC}"
        rm -f "${SCRIPT_DIR}/bot.pid"
        exit 0
    fi
    sleep 1
done

# If the process is still running, force kill it
echo -e "${RED}Bot didn't stop gracefully. Forcing termination...${NC}"
kill -9 $BOT_PID 2>/dev/null

# Check if the kill was successful
if ! ps -p $BOT_PID > /dev/null; then
    echo -e "${GREEN}Bot has been terminated.${NC}"
    rm -f "${SCRIPT_DIR}/bot.pid"
else
    echo -e "${RED}Failed to terminate bot. Please check process manually.${NC}"
fi