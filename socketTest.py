import socket

# Set up a socket to listen for incoming messages
host = "0.0.0.0"
port = 50000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((host, port))
    server_socket.listen()

    print(f"Listening for messages on {host}:{port}")

    while True:
        client_socket, _ = server_socket.accept()
        data = client_socket.recv(1024).decode("utf-8")
        print(f"Received message: {data}")
        # Process the incoming data here