#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

"""
import csv
import os
import time
from Ironman import predict
from Ironman.Sender import Sender


host = 'CN0005DPW12015.dir.slb.com'
queue = 'hello'
exchange = 'coding_league_answer'
routing_key = '28d08bbb-bd3a-4ea2-9ca9-b67d030c67c1'
static_model_file = r"\\cn0005dpdev075\CodeLeague\ironman1.pkl".replace('\\', '/')


folder = r"\\cn0005dpdev075\CodeLeague\Ironman\20191217".replace('\\', '/')
s = Sender(host, queue, exchange, routing_key)

# for i in os.listdir(folder):
i = "77"
print(f'task id:{i}')
image_folder2 = os.path.join(folder, i)
csv_file = os.path.join(image_folder2, "img/predict1.csv")
if os.path.exists(csv_file):
    with open(csv_file) as f:
        res = f.read()
        res = res.split('\n')
        print(len(res))
        if len(res) == 51:
            res = "Dataset" + i +":" + ','.join(res)
            # print(res)
            s.publish(res)