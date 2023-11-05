#!/bin/bash
# Raspberry Pi SSH connection details
PI_USERNAME="lightwork"
PI_HOST="raspberrypi.local"
PI_PASSWORD="TEAM10"
export DISPLAY=:0
# Execute the SSH connection and script execution
plink.exe -ssh -t $PI_USERNAME@$PI_HOST -pw $PI_PASSWORD "export DISPLAY=:0; cd /home/lightwork/scripts/Structured_Light_Camera; libcamerify python3 GrayCoding.py" 
if [ $? -eq 0 ]; then
    exit 0
    echo "Script ran successfully (Exit code: $?)"
else
    exit 2
    echo "Script encountered an error (Exit code: $?)"
fi
