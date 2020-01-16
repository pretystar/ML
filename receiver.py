from src import message_queue

sender = message_queue.Receiver()
sender.start()
print('send message')
# sender.close()
