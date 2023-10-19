import tkinter as tk
import subprocess
import threading
import os
import signal
import time

def start_script():
    global script_process
    script_path = 'connectToPi.sh'
    # script_process = subprocess.Popen(['python', 'test.py'])
    script_process = subprocess.Popen("plink lightwork@raspberrypi.local -pw TEAM10",shell=False,stdin=subprocess.PIPE)
    print("Script executed successfully")
    print("Script output:")
    print(script_process.stdout)
    script_process.stdin.write("cd scripts\n".encode('utf-8'))
    script_process.stdin.write("libcamerify python3 image.py\n".encode('utf-8'))
    script_process.stdin.close()
    output, errors = script_process.communicate()

    # Print the output and errors from the SSH session
    print("Output:")
    print(output)

    print("Errors:")
    print(errors)



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

script_process = None

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