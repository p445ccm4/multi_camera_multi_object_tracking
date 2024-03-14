#TODO: Read in X_w, Y_w, feature from local
#TODO: Compare them with existing tracks, match or create a new track

import socket
import pickle

# Create a socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the address and port
s.bind(('127.0.0.1', 5000))

# Listen for incoming connections
s.listen(5)

# Accept a connection
conn, addr = s.accept()

while True:
    # Receive the serialized object from the sender
    data = b''
    while True:
        chunk = conn.recv(1024)
        if not chunk:
            break
        data += chunk
    # Deserialize the object using pickle
    obj = pickle.loads(data)

    # Print the received object
    print(obj)

# Close the socket
s.close()
