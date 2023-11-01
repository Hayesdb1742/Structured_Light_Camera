import tkinter as tk
import subprocess
import threading
import os
import signal
import time

def start_script():
    global script_process
    script_path = './connectToPi.sh'
    # script_process = subprocess.Popen(['python', 'test.py'])
    script_process = subprocess.run(['bash', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if script_process.returncode == 0:
        #successful connection
        start_button.config(state=tk.DISABLED, bg='green')
        pass
    else:
        #unsuccessful connection
        start_button.config(state=tk.DISABLED, bg='red')
        pass

    terminate_button.config(state=tk.NORMAL, bg='lightgray')
# Function to start the next script in the sequence.
def start_next_script():
    global script_process, current_script, scripts_to_run
    # Check if the script sequence is still running and there are remaining scripts.
    # if is_running and current_script < len(scripts_to_run):
    script_name = scripts_to_run[0]  # Get the current script name.
        # Start the current script.
    # script_process = ['sh', './test.sh']
    current_script += 1  # Move to the next script in the sequence.
    # Update button states.
    start_button.config(state=tk.DISABLED, bg='darkgray')
    terminate_button.config(state=tk.NORMAL, bg='lightgray')

def terminate_script():
    if script_process.poll() is None:
        script_process.terminate()
    start_button.config(state=tk.NORMAL, bg='lightgray')
    terminate_button.config(state=tk.DISABLED, bg='darkgray')

def on_closing():
    if script_process is not None:
        script_process.terminate()
    root.destroy()

def check_script_status():
    while True:
        if script_process is not None:
            start_button.config(state=tk.DISABLED, bg='darkgray')
            terminate_button.config(state=tk.NORMAL, bg='lightgray')
            status_label.config(text="Program is running" + "." * (int(time.time()) % 4))
        else:
            start_button.config(state=tk.NORMAL, bg='lightgray')
            terminate_button.config(state=tk.DISABLED, bg='darkgray')
            status_label.config(text="Program is not running")
        root.update()
        time.sleep(1)  # Update every second

root = tk.Tk()
root.title("Script Controller")

initial_width = 300
initial_height = 200
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - initial_width) // 2
y = (screen_height - initial_height) // 2
root.geometry(f"{initial_width}x{initial_height}+{x}+{y}")

# List of scripts to be run in sequence. The names below are placeholders, can be replaced by actual file names later.
# If more than three file executions are linked together, add the file names in their execution sequence as needed.
scripts_to_run = ['connectToPi.sh']
current_script = 0  # Index to keep track of the current script.
script_process = None  # Will hold the subprocess for the current script.
is_running = False  # Flag to determine if the script sequence is currently running.

button_padx = 20
button_pady = 10

start_button = tk.Button(root, text="Connect", command=start_script, bg='lightgray')
start_button.pack(fill=tk.BOTH, expand=True, padx=button_padx, pady=button_pady)
start_button.pack_propagate(0)

terminate_button = tk.Button(root, text="Terminate", command=terminate_script, bg='darkgray', state=tk.DISABLED)
terminate_button.pack(fill=tk.BOTH, expand=True, padx=button_padx, pady=button_pady)
terminate_button.pack_propagate(0)

status_label = tk.Label(root, text="Program is not running", font=("Helvetica", 12))
status_label.pack(pady=10)

status_checker_thread = threading.Thread(target=check_script_status)
status_checker_thread.daemon = True
status_checker_thread.start()

root.protocol("WM_DELETE_WINDOW", on_closing)  # Handle window closing event

root.mainloop()