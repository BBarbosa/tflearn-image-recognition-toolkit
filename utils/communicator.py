import socket
import sys
from colorama import init
from termcolor import colored

# init colored print
init()

if (len(sys.argv) < 4):
    print(colored("Call: $ python %s {ip} {port} {image_path} [class]" % sys.argv[0],"red"))
    sys.exit(colored("ERROR: Not enough arguments!","red"))

ip         = sys.argv[1]
port       = int(sys.argv[2])  
image_path = sys.argv[3]

try:
    classid = sys.argv[4]
except:
    classid = None
    pass

clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect((ip, port))                    # connects to local server (see classify_sv.py)

message = image_path
if(classid is not None): message += " " + classid   # assumes that is the only space to split data further

print("[INFO] Send message:", message)
clientsocket.send(bytes(message, 'ascii'))          # converts to byte and send to server
clientsocket.settimeout(3)                          # set a timeout of 3 seconds

while True:
    try:
        response = clientsocket.recv(512)                   # receive message with a maximum size of 512 bytes
        response = str(response.decode('ascii'))            # decode received message to string format

        if(len(response) == 0):
            break

        print("[INFO] Received message:", response)
    except Exception as e:
        print("[EXCEPTION]", e)
        break

clientsocket.shutdown(1)
clientsocket.close()
print("[INFO] All done!")