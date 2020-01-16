import pika

host = 'localhost'
queue = 'hello'
exchange = ''
routing_key = 'hello'


class Receiver:
    def __init__(self, callback):
        credential = pika.credentials.PlainCredentials("taiji", "taiji")
        parameters = pika.ConnectionParameters(host=host,credential=credential)
        self.connection = pika.BlockingConnection(parameters=parameters)
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=queue)
        self.channel.basic_consume(
            queue=queue,
            on_message_callback=lambda ch, method, properties, body: callback(body),
            auto_ack=True)

    def start(self):
        self.channel.start_consuming()

    def stop(self):
        self.connection.close()


class Sender:
    def __init__(self):
        credential = pika.credentials.PlainCredentials("taiji", "taiji")
        parameters = pika.ConnectionParameters(host=host,credential=credential)
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=queue)

    def publish(self, message):
        self.channel.basic_publish(
            exchange=exchange,
            routing_key=routing_key,
            body=message)
        print(' [x] Sent "{}"'.format(message))

    def close(self):
        self.connection.close()


if __name__ == '__main__':
    sender = Sender()
    sender.publish('cbj')
    print('send message')
    sender.close()
