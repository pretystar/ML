from src import processor
from src import web

handler = processor.Handler()
server = web.Server(8888, handler)
try:
    print('tornado server starts.')
    server.start()
except KeyboardInterrupt:
    print('tornado server stops.')
    server.stop()
    exit(0)