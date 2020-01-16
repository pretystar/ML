import threading
from src import processor
from src import message_queue
from src import web

PATH = None

handler = processor.Handler(PATH)
receiver = message_queue.Receiver(handler.callback)
server = web.Server(8888, handler)

for target in (receiver.start, server.start):
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()

print('services running, press ctrl+c to stop')
while True:
    input('')
