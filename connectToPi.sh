#!/bin/bash
# Raspberry Pi SSH connection details
PI_USERNAME="lightwork"
PI_HOST="raspberrypi.local"
PI_PASSWORD="TEAM10"
export DISPLAY=:0

# Execute the SSH connection and script execution
plink -ssh $PI_USERNAME@$PI_HOST -pw $PI_PASSWORD "export DISPLAY=:0; cd /home/lightwork/scripts/Structured_Light_Camera/ProjectorCameraCalibration; /bin/bash"


if [ $? -eq 0 ]; then
    echo "Script ran successfully"
    (exit 0)
else
    (exit 2)
    echo "/nScript encountered an error (Exit code: $?)"
fi
