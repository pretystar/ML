from src import processor
from src import web

PATH = None

handler = processor.Handler(PATH)
server = web.Server(8889, handler)
server.start()