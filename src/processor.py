from . import learning

class Handler:
    def __init__(self, path):
        self.message = ''
        if path is not None:
            self.model = learning.load_model(path)

    def callback(self, message_body):
        self.message = message_body.decode('ascii')
        print('ReceivedMessageHandler gets message {}'.format(self.message))
