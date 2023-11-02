#!/bin/bash
# Raspberry Pi SSH connection details
PI_USERNAME="bentley.206"
PI_HOST="rh050.coeit.osu.edu"
PI_PASSWORD="Jorge1742!!!"
export DISPLAY=:0

# Execute the SSH connection and script execution
plink -ssh $PI_USERNAME@$PI_HOST -pw $PI_PASSWORD "ls -l"
echo "plink run $PI_USERNAME"

if [ $? -eq 0 ]; then
    echo "Script ran successfully"
    (exit 0)
else
    (exit 2)
    echo "Script encountered an error (Exit code: $?)"
fi
