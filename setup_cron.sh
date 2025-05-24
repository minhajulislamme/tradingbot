#!/bin/bash

# Setup cron job to start the bot after system reboot
# and to perform regular backups

# Get the absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up cron jobs for automatic startup and backups...${NC}"

# Create a temporary file for crontab
TEMP_CRON=$(mktemp)

# Export existing crontab
crontab -l > "$TEMP_CRON" 2>/dev/null || echo "# Binance Trading Bot crontab" > "$TEMP_CRON"

# Check if entries already exist
if grep -q "binancebot" "$TEMP_CRON"; then
    echo -e "${YELLOW}Cron jobs for Binance Trading Bot already exist. Updating...${NC}"
    # Remove existing entries
    sed -i '/binancebot/d' "$TEMP_CRON"
fi

# Add entries for reboot and daily backup
cat << EOF >> "$TEMP_CRON"
# Binance Trading Bot - Start after reboot (wait 60 seconds for network)
@reboot sleep 60 && cd $SCRIPT_DIR && ./start_bot.sh >> $SCRIPT_DIR/logs/cron_startup.log 2>&1

# Binance Trading Bot - Daily backup at 00:00
0 0 * * * cd $SCRIPT_DIR && ./backup_data.sh >> $SCRIPT_DIR/logs/backup.log 2>&1
EOF

# Install new crontab
crontab "$TEMP_CRON"
rm "$TEMP_CRON"

echo -e "${GREEN}Cron jobs installed successfully!${NC}"
echo -e "${GREEN}The bot will now automatically:${NC}"
echo -e "${GREEN}- Start after system reboots${NC}"
echo -e "${GREEN}- Perform daily backups at midnight${NC}"