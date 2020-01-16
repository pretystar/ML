from src import message_queue

sender = message_queue.Sender()
sender.publish('cbj')
print('send message')
sender.close()
