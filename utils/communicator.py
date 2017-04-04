import socket,sys
from colorama import init
from termcolor import colored

# init colored print
init()

if (len(sys.argv) < 3):
    print(colored("Call: $ python communicator.py {image_path} {class}","red"))
    sys.exit(colored("ERROR: Not enough arguments!","red"))

clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect(('localhost', 8090))       # connects to local server (see classify_sv.py)
message = sys.argv[1] + " " + sys.argv[2]       # assumes that is the only space to split data further
clientsocket.send(bytes(message, 'ascii'))      # converts to byte and send to server