#! /bin/python3
################################################################################
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

from threading import Thread
import queue
import sys

from generation import matmul

import plotter
from plotter import heatMap2D
from plotter import heatMap3D
from plotter import scatter3D


from reporter import clear_screen
from reporter import summaryStats

from metrics import Absolute
from metrics import SampleRelative

from metrics import Flops
from metrics import Bandwidth


import time

class Sample:
    def __init__(self, sample_line):
        _, name, prb, flops, bandwidth = sample_line.strip().split(",")
        self.name = name
        self.primitive = Primitive.from_repro(prb)
        self.flops = float(flops)
        self.bandwidth = float(bandwidth)
    def kind(self):
        if self.name != "":
            return self.name + ": " + str(self.primitive.kind)
        else:
            return str(self.primitive.kind)

ExitToken = "Exit"
def ingest_data(sample_queue):
    for line in sys.stdin:
        if line.startswith("sample"):
            sample_queue.put(Sample(line))
    sample_queue.put(ExitToken)

def reporter(sample_queue):
    reports = [
        summaryStats(SampleRelative(), Bandwidth())
    ]
    last_report_update = time.monotonic()
    has_plot = False
    for r in reports:
        if isinstance(r, plotter.Plot):
            has_plot = True
            break

    if(has_plot):
        plotter.initialize()

    sample = None
    no_data = True
    while sample != ExitToken:
        try:
            has_new_data = False
            while True:
                sample = sample_queue.get(block=no_data, timeout=0.01)
                if sample == ExitToken:
                    break
                for r in reports:
                    r.add(sample)
                has_new_data = True
                no_data = False
        except queue.Empty:
            pass

        update_time = time.monotonic()
        if(has_new_data and (update_time - last_report_update > 1)):
            clear_screen()
            last_report_update = update_time
            for r in reports:
                r.update()
            last_update = time.monotonic()

        if(has_plot):
            plotter.update()

    if(has_plot):
        plotter.finalize()

if __name__ == "__main__":
    sample_queue = queue.Queue()
    producer = Thread(target=ingest_data, args=(sample_queue,))
    producer.start()
    reporter(sample_queue)
    producer.join()
