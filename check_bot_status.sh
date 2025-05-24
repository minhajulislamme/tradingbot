#!/bin/bash

# Check the status of the Binance Trading Bot
# This script shows the bot status and recent logs

# Get the absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if the bot is running
if [ -f "${SCRIPT_DIR}/bot.pid" ]; then
    BOT_PID=$(cat "${SCRIPT_DIR}/bot.pid")
    if ps -p $BOT_PID > /dev/null; then
        echo -e "${GREEN}Bot is running with PID ${BOT_PID}${NC}"
        echo -e "${YELLOW}Runtime: $(ps -o etime= -p $BOT_PID)${NC}"
        
        # Show memory usage
        MEM_USAGE=$(ps -o rss= -p $BOT_PID)
        MEM_USAGE_MB=$(( $MEM_USAGE / 1024 ))
        echo -e "${YELLOW}Memory usage: ${MEM_USAGE_MB} MB${NC}"
    else
        echo -e "${RED}Found bot.pid file but process ${BOT_PID} is not running${NC}"
        echo -e "${YELLOW}The bot may have crashed or been terminated improperly${NC}"
    fi
else
    # Try to find the bot process even without pid file
    BOT_PID=$(pgrep -f "python3 ${SCRIPT_DIR}/main.py")
    if [ -n "$BOT_PID" ]; then
        echo -e "${GREEN}Bot is running with PID ${BOT_PID} (PID file missing)${NC}"
        echo -e "${YELLOW}Runtime: $(ps -o etime= -p $BOT_PID)${NC}"
        
        # Save the found PID to the pid file
        echo $BOT_PID > "${SCRIPT_DIR}/bot.pid"
        echo -e "${GREEN}Created new bot.pid file${NC}"
    else
        echo -e "${RED}Bot is not running${NC}"
    fi
fi

# Display the latest log entries
echo ""
echo -e "${CYAN}=== Latest Bot Log Entries ===${NC}"
LOG_FILE="${SCRIPT_DIR}/logs/bot_console.log"
if [ -f "$LOG_FILE" ]; then
    echo -e "${YELLOW}Last modified: $(stat -c '%y' $LOG_FILE)${NC}"
    echo ""
    tail -n 30 "$LOG_FILE"
else
    echo -e "${RED}No log file found at ${LOG_FILE}${NC}"
fi

# Display latest trading log if available
echo ""
echo -e "${CYAN}=== Latest Trading Log Entries ===${NC}"
LATEST_LOG=$(find "${SCRIPT_DIR}/logs" -name "trading_bot_*.log" -type f -printf "%T@ %p\n" | sort -nr | head -n1 | cut -d' ' -f2-)
if [ -n "$LATEST_LOG" ]; then
    echo -e "${YELLOW}Log file: ${LATEST_LOG}${NC}"
    echo -e "${YELLOW}Last modified: $(stat -c '%y' "$LATEST_LOG")${NC}"
    echo ""
    tail -n 30 "$LATEST_LOG"
else
    echo -e "${RED}No trading log files found${NC}"
fi