#!/bin/bash

# Binance Trading Bot Setup Script
# This script sets up the trading bot with a virtual environment
# and configures it to run as a systemd service for 24/7 operation

set -e  # Exit immediately if a command exits with non-zero status

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BOT_DIR=$SCRIPT_DIR
VENV_DIR="${BOT_DIR}/venv"
SERVICE_NAME="binancebot"

# Function to display messages with color
print_message() {
    echo -e "${2:-$GREEN}$1${NC}"
}

# Check if script is run as root
if [ "$EUID" -eq 0 ]; then
    print_message "⚠️  This script should not be run as root to avoid permission issues." "$YELLOW"
    print_message "Please run it as a regular user with sudo privileges." "$YELLOW"
    exit 1
fi

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    print_message "❌ Python 3 is not installed. Please install Python 3.8+ and try again." "$RED"
    exit 1
fi

# Install required system packages if not already installed
print_message "📦 Checking and installing required system packages..."
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip supervisor

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    print_message "🔧 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    print_message "✅ Virtual environment already exists." "$YELLOW"
fi

# Activate virtual environment and install requirements
print_message "📚 Installing dependencies..."
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install -r "${BOT_DIR}/requirements.txt"

# Create necessary directories
print_message "📁 Creating necessary directories..."
mkdir -p "${BOT_DIR}/logs"
mkdir -p "${BOT_DIR}/state"
mkdir -p "${BOT_DIR}/reports"

# Set up supervisor configuration
print_message "🔧 Setting up supervisor configuration..."
SUPERVISOR_CONF="/etc/supervisor/conf.d/${SERVICE_NAME}.conf"

sudo tee "$SUPERVISOR_CONF" > /dev/null << EOF
[program:${SERVICE_NAME}]
directory=${BOT_DIR}
command=${VENV_DIR}/bin/python3 ${BOT_DIR}/main.py
user=$(whoami)
autostart=true
autorestart=true
startsecs=10
startretries=3
stopwaitsecs=300
stdout_logfile=${BOT_DIR}/logs/supervisor_stdout.log
stderr_logfile=${BOT_DIR}/logs/supervisor_stderr.log
environment=PATH="${VENV_DIR}/bin:%(ENV_PATH)s"
EOF

# Create start and stop scripts
print_message "📝 Creating convenient start/stop scripts..."

# Create start script
cat > "${BOT_DIR}/start_bot.sh" << EOF
#!/bin/bash
echo "Starting Binance Trading Bot..."
sudo supervisorctl start ${SERVICE_NAME}
sudo supervisorctl status ${SERVICE_NAME}
EOF
chmod +x "${BOT_DIR}/start_bot.sh"

# Create stop script
cat > "${BOT_DIR}/stop_bot.sh" << EOF
#!/bin/bash
echo "Stopping Binance Trading Bot..."
sudo supervisorctl stop ${SERVICE_NAME}
sudo supervisorctl status ${SERVICE_NAME}
EOF
chmod +x "${BOT_DIR}/stop_bot.sh"

# Create status check script
cat > "${BOT_DIR}/check_status.sh" << EOF
#!/bin/bash
echo "Checking Binance Trading Bot status..."
sudo supervisorctl status ${SERVICE_NAME}
echo ""
echo "Last 20 log lines:"
tail -n 20 ${BOT_DIR}/logs/supervisor_stdout.log
EOF
chmod +x "${BOT_DIR}/check_status.sh"

# Create a backup script
cat > "${BOT_DIR}/backup_data.sh" << EOF
#!/bin/bash
BACKUP_DIR="\${HOME}/binancebot_backups"
TIMESTAMP=\$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="\${BACKUP_DIR}/binancebot_backup_\${TIMESTAMP}.tar.gz"

mkdir -p "\${BACKUP_DIR}"

echo "Creating backup of Binance Trading Bot data..."
tar -czf "\${BACKUP_FILE}" -C "${BOT_DIR}" state reports logs

echo "Backup created at: \${BACKUP_FILE}"
echo "Removing backups older than 7 days..."
find "\${BACKUP_DIR}" -name "binancebot_backup_*.tar.gz" -type f -mtime +7 -delete
EOF
chmod +x "${BOT_DIR}/backup_data.sh"

# Reload supervisor configuration
print_message "🔄 Reloading supervisor configuration..."
sudo supervisorctl update
sudo supervisorctl reread

print_message "✅ Setup completed successfully!"
print_message "📊 To start the bot: ${BOT_DIR}/start_bot.sh"
print_message "🛑 To stop the bot: ${BOT_DIR}/stop_bot.sh"
print_message "ℹ️  To check status: ${BOT_DIR}/check_status.sh"
print_message "💾 To backup data: ${BOT_DIR}/backup_data.sh"

# Display status
echo ""
print_message "Current bot status:"
sudo supervisorctl status ${SERVICE_NAME}