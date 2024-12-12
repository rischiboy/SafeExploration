#!/bin/bash

log_dir="run/logs/"
base_name="Pendulum_trainer"
extension=".log"
counter=0

timestamp=$(date +"%Y-%m-%dT%H:%M")

cd "$log_dir" || exit

# Check for existing log files and find the next available increment
while [[ -e "${base_name}_${counter}${extension}" ]]; do
    ((counter++))
done

next_file="${base_name}_${counter}_${timestamp}${extension}"
echo "Next available log file: $next_file"