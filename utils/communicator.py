import socket,sys
from colorama import init
from termcolor import colored

# init colored print
init()

if (len(sys.argv) < 2):
    print(colored("Call: $ python communicator.py {ip} {port} {image_path} {class}","red"))
    sys.exit(colored("ERROR: Not enough arguments!","red"))

ip         = sys.argv[1]
port       = int(sys.argv[2])  
image_path = sys.argv[3]
classid    = sys.argv[4]

clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect((ip, port))                    # connects to local server (see classify_sv.py)
message = image_path + " " + classid                # assumes that is the only space to split data further
clientsocket.send(bytes(message, 'ascii'))          # converts to byte and send to server