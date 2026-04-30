# Function to run a command and wait for a specific phrase
run_and_wait() {
    local cmd="$1"
    local phrase="$2"
    local log_file
    log_file=$(mktemp)
    
    # Start command in a new terminal with logging
    gnome-terminal -- bash -c "eval $cmd | tee \"$log_file\"; exec bash"
    
    # Wait for the phrase to appear in the log
    echo "Waiting for '$phrase'..."
    tail -f -n 0 "$log_file" | grep -m 1 "$phrase" >/dev/null
    
    # Clean up the log file
    rm -f "$log_file"
}

# Cleanup function to stop all containers
cleanup() {
    echo "Stopping all Docker containers..."
    docker stop $(docker ps -q) 2>/dev/null
    echo "Closing all terminal windows..."
    # Close all terminal windows except the original one
    pgrep -f "gnome-terminal" | grep -v $$ | xargs kill -9 2>/dev/null
    
    # Close the original terminal window (this will terminate the script)
    kill -9 $PPID 2>/dev/null

    echo "Cleanup complete."
}

# Register cleanup to run on script exit
trap cleanup EXIT

# Wait for Docker to be ready
while ! docker info >/dev/null 2>&1; do
    echo "Waiting for Docker to start..."
    sleep 5
done

# Command sequences with their corresponding wait phrases
# Format: (command phrase) pairs
commands=(
    "cd ~/prova/Vision_Docker_Orrbec && docker compose up" "OBCameraIntrinsic"
    "cd ~/prova/Supreme-Docker-qb-Integration && docker compose up" "Basic environment generated."
)

# Run commands sequentially
for ((i=0; i<${#commands[@]}; i+=2)); do
    run_and_wait "${commands[i]}" "${commands[i+1]}"
done

echo "All services started successfully! Starting task..."

docker exec supreme-docker-qb-integration-pnrr_project_jazzy-1 bash -c 'source home/blah/workspace/.bash_sources && ros2 service call /start_trigger std_srvs/srv/Empty'

