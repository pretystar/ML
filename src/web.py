import tornado.ioloop
import tornado.web
from . import processor
import json

class MainHandler(tornado.web.RequestHandler):
    def initialize(self, handler):
        self.handler = handler

    def get(self):
        self.write('Hello, {}'.format(self.handler.message))

    def post(self):
        pass
        # param = self.request.body.decode('utf-8')
        # prarm = json.loads(param)
        # print(param)

class Step2Validation(tornado.web.RequestHandler):
    def get(self):
        self.write('58')

class Step2Evaluation(tornado.web.RequestHandler):
    def get(self):
        self.write('')

class Server:
    def __init__(self, port, handler):
        self.handler = handler
        self.app = self.__make_app()
        self.app.listen(port)
        self.ioloop = tornado.ioloop.IOLoop.current()

    def start(self):
        self.ioloop.start()

    def stop(self):
        self.ioloop.stop()

    def __make_app(self):
        return tornado.web.Application([
            (r'/', MainHandler, dict(handler=self.handler)),
            (r'/step2valiadation', Step2Validation),
            (r'/step2evaluation', Step2Evaluation)
        ])


