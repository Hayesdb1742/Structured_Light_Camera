import tkinter as tk
import subprocess
import threading
import time

# Function to handle starting the script sequence.
def start_script():
    global script_process, current_script, scripts_to_run, is_running
    # Check if the script sequence is not already running.
    if not is_running:
        current_script = 0  # Reset to the first script
        is_running = True  # Set the script sequence as running.
        start_next_script()  # Start the first script.

# Function to start the next script in the sequence.
def start_next_script():
    global script_process, current_script, scripts_to_run
    # Check if the script sequence is still running and there are remaining scripts.
    if is_running and current_script < len(scripts_to_run):
        script_name = scripts_to_run[current_script]  # Get the current script name.
        # Start the current script.
        script_process = subprocess.Popen(['python', script_name])
        current_script += 1  # Move to the next script in the sequence.
        # Update button states.
        start_button.config(state=tk.DISABLED, bg='darkgray')
        terminate_button.config(state=tk.NORMAL, bg='lightgray')

# Function to terminate the currently running script.
def terminate_script():
    global script_process, is_running
    is_running = False  # Set the script sequence as not running to prevent further scripts from starting.
    # Check if there is a running script to terminate.
    if script_process is not None and script_process.poll() is None:
        script_process.terminate()  # Terminate the current script.
    # Update button states.
    start_button.config(state=tk.NORMAL, bg='lightgray')
    terminate_button.config(state=tk.DISABLED, bg='darkgray')

# Function to handle the event when the user tries to close the application window.
def on_closing():
    global is_running
    is_running = False  # Set the script sequence as not running.
    # Check if there is a running script to terminate.
    if script_process is not None and script_process.poll() is None:
        script_process.terminate()  # Terminate the running script.
    root.destroy()  # Close the application.

# Function to check and update the status of scripts, runs in a separate thread.
def check_script_status():
    global script_process, current_script, scripts_to_run, is_running
    while True:  # This loop will keep checking the script status.
        # Check if a script is currently running.
        if script_process is not None and script_process.poll() is None:
            # Script is running.
            script_name = scripts_to_run[current_script - 1]  # Get the current script name.
            # Update the status label with the current script's name.
            status_label.config(text=f"{script_name} is running" + "." * (int(time.time()) % 4))
        else:
            # If the current script has finished and there are more scripts to run.
            if is_running and current_script < len(scripts_to_run):
                start_next_script()  # Start the next script.
            else:
                # No scripts are running or remaining.
                start_button.config(state=tk.NORMAL, bg='lightgray')
                terminate_button.config(state=tk.DISABLED, bg='darkgray')
                status_label.config(text="No script is running")  # Update the status label.

        root.update()  # Update the GUI elements.
        time.sleep(1)  # Wait for 1 second before the next check.

# Set up the main application window.
root = tk.Tk()
root.title("Script Controller")

# Set the initial dimensions and position of the application window.
initial_width = 300
initial_height = 200
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - initial_width) // 2
y = (screen_height - initial_height) // 2
root.geometry(f"{initial_width}x{initial_height}+{x}+{y}")

# List of scripts to be run in sequence. The names below are placeholders, can be replaced by actual file names later.
# If more than three file executions are linked together, add the file names in their execution sequence as needed.
scripts_to_run = ['file_name_1.py', 'file_name_2.py', 'file_name_3.py']
current_script = 0  # Index to keep track of the current script.
script_process = None  # Will hold the subprocess for the current script.
is_running = False  # Flag to determine if the script sequence is currently running.

# GUI setup for buttons and status label.
button_padx = 20
button_pady = 10

# "Start" button to start the script sequence.
start_button = tk.Button(root, text="Start", command=start_script, bg='lightgray')
start_button.pack(fill=tk.BOTH, expand=True, padx=button_padx, pady=button_pady)
start_button.pack_propagate(0)

# "Terminate" button to stop the currently running script.
terminate_button = tk.Button(root, text="Terminate", command=terminate_script, bg='darkgray', state=tk.DISABLED)
terminate_button.pack(fill=tk.BOTH, expand=True, padx=button_padx, pady=button_pady)
terminate_button.pack_propagate(0)

# Label to display the current script status.
status_label = tk.Label(root, text="No script is running", font=("Helvetica", 12))
status_label.pack(pady=10)

# Start a new thread to check the script status periodically.
status_checker_thread = threading.Thread(target=check_script_status)
status_checker_thread.daemon = True  # This thread will close when the main application closes.
status_checker_thread.start()

# Assign a function to handle the window closing event.
root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the GUI event loop.
root.mainloop()
