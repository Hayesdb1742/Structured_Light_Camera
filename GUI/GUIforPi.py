import tkinter as tk
import paramiko

# Initialize global variables
ssh_client = None
is_connected = False
is_running = False
scripts_to_run = ['file_name_1.py', 'file_name_2.py', 'file_name_3.py']
current_script = 0

# GUI Functions
def connect_to_pi():
    global ssh_client, is_connected
    MY_USERNAME = "lightwork"  # Your SSH username
    MY_HOST = "raspberrypi.local"  # Your SSH host
    MY_PASSWORD = "TEAM10"  # Your SSH password

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh_client.connect(MY_HOST, username=MY_USERNAME, password=MY_PASSWORD)
        is_connected = True
        connect_button.config(bg='green', text="Connected")
        connection_status_label.config(text="Connected to Raspberry Pi")
        start_button.config(state=tk.NORMAL)
        disconnect_button.config(state=tk.NORMAL)
        terminate_button.config(state=tk.NORMAL)  # Enable the terminate button upon connection
    except paramiko.AuthenticationException:
        is_connected = False
        connect_button.config(bg='red', text="Connect", state=tk.NORMAL)
        connection_status_label.config(text="Authentication failed.")
    except paramiko.SSHException as e:
        is_connected = False
        connect_button.config(bg='red', text="Connect", state=tk.NORMAL)
        connection_status_label.config(text=f"SSH connection error: {e}")

def disconnect_from_pi():
    global ssh_client, is_connected
    if ssh_client is not None:
        ssh_client.close()
    is_connected = False
    connect_button.config(bg='lightgray', text="Connect", state=tk.NORMAL)
    connection_status_label.config(text="Disconnected from Raspberry Pi")
    start_button.config(state=tk.DISABLED)
    disconnect_button.config(state=tk.DISABLED)
    terminate_button.config(state=tk.DISABLED)  # Disable the terminate button upon disconnection

def execute_script(script_name):
    global ssh_client
    if is_connected:
        # Change the directory here
        stdin, stdout, stderr = ssh_client.exec_command(f'python3 /home/lightwork/scripts/Structured_Light_Camera/{script_name}')
        return stdout.read().decode('utf-8'), stderr.read().decode('utf-8')
    return None, None

def start_script():
    global is_running, current_script
    if is_connected and not is_running:
        is_running = True
        current_script = 0
        start_next_script()
        terminate_button.config(state=tk.NORMAL)  # Enable the terminate button when script starts

def start_next_script():
    global current_script, scripts_to_run
    if current_script < len(scripts_to_run):
        script_name = scripts_to_run[current_script]
        output, error = execute_script(script_name)
        if output or error:
            execution_output_label.config(text=f"Output from {script_name}:\n{output}\nErrors:\n{error}")
            current_script += 1
        else:
            execution_status_label.config(text="Failed to execute script.")
            terminate_script()
        if current_script < len(scripts_to_run):
            root.after(1000, start_next_script)  # Proceed to next script after 1 second
        else:
            transfer_stl()

def terminate_script():
    global is_running, current_script
    is_running = False
    current_script = 0
    execution_status_label.config(text="Script terminated")
    start_button.config(state=tk.NORMAL)
    terminate_button.config(state=tk.DISABLED)  # Disable the terminate button when script is terminated

def transfer_stl():
    global ssh_client
    sftp_client = ssh_client.open_sftp()
    # Change the directory here
    sftp_client.get('/home/lightwork/scripts/Structured_Light_Camera/output.stl', 'output.stl')
    sftp_client.close()
    execution_status_label.config(text=".STL file transferred successfully!")

# GUI Layout
root = tk.Tk()
root.title("Pi Script Controller")

# Set a minimum button width and padding
button_width = 15
padx = 10
pady = 10

# Frames
connection_frame = tk.Frame(root, padx=padx, pady=pady)
connection_frame.grid(row=0, column=0, sticky="nsew")

execution_frame = tk.Frame(root, padx=padx, pady=pady)
execution_frame.grid(row=0, column=1, sticky="nsew")

# Buttons
connect_button = tk.Button(connection_frame, text="Connect", bg='lightgray', command=connect_to_pi, width=button_width)
connect_button.grid(row=0, column=0, sticky="ew", padx=padx, pady=pady)

disconnect_button = tk.Button(connection_frame, text="Disconnect", bg='lightgray', command=disconnect_from_pi, state=tk.DISABLED, width=button_width)
disconnect_button.grid(row=1, column=0, sticky="ew", padx=padx, pady=pady)

connection_status_label = tk.Label(connection_frame, text="Not connected", font=("Helvetica", 10))
connection_status_label.grid(row=2, column=0, sticky="nsew")

start_button = tk.Button(execution_frame, text="Start", bg='lightgray', command=start_script, state=tk.DISABLED, width=button_width)
start_button.grid(row=0, column=0, sticky="ew", padx=padx, pady=pady)

terminate_button = tk.Button(execution_frame, text="Terminate", bg='darkgray', command=terminate_script, state=tk.DISABLED, width=button_width)
terminate_button.grid(row=1, column=0, sticky="ew", padx=padx, pady=pady)

execution_status_label = tk.Label(execution_frame, text="No script is running", font=("Helvetica", 10))
execution_status_label.grid(row=2, column=0, sticky="nsew")

execution_output_label = tk.Label(execution_frame, text="Script output will appear here", font=("Helvetica", 10))
execution_output_label.grid(row=3, column=0, sticky="nsew")

# Configure the weight of rows and columns for the grid
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1, uniform="group1")
root.grid_columnconfigure(1, weight=1, uniform="group1")

# Run the main loop
root.mainloop()
