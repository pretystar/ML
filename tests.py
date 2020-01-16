import unittest
from src import *


class MessageProcessorTestSuite(unittest.TestCase):
    def test_should_create_received_message_handler(self):
        received_message_handler = processor.Handler()
        assert received_message_handler is not None


if __name__ == '__main__':
    unittest.main()
