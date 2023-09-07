#!/bin/bash

# Raspberry Pi SSH connection details
PI_USERNAME="lightwork"
PI_HOST="raspberrypi.local"
PI_PASSWORD="TEAM10"

export DISPLAY=:0.0

# Path to the shell script to execute on the Raspberry Pi
REMOTE_SCRIPT="home/lightwork/scripts/imageControl.sh"

# Execute the SSH connection and script execution
plink -ssh -X $PI_USERNAME@$PI_HOST -pw $PI_PASSWORD
