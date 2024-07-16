
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

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mticker
import numpy

import metrics
from generation import matmul

from metrics import perf_data

#Protects against division by zero
def eff(value, max_value):
    if max_value == 0:
        return 1
    else:
        return value / max_value

#Helpers for workaround for https: //github.com/matplotlib/matplotlib/issues/209
def log_tick_formatter(val, pos=None):
    return "{:.0f}".format(2**val)


def rescale(data):
    return numpy.log2(data)


class Plot:
    def __init__(self, proj="3d"):
        self.fig = plt.gcf()
        subplots = self.fig.get_axes()
        cols = len(subplots) + 1
        gs = gridspec.GridSpec(1, cols)
#Update the existing subplot positions
        for i, ax in enumerate(subplots):
            ax.set_position(gs[i].get_position(self.fig))
            ax.set_subplotspec(gs[i])

#Add the new subplot
        self.ax = self.fig.add_subplot(gs[cols - 1], projection=proj)

class heatMap2D(Plot):
    def __init__(self, x_label, y_label, scaling, metricValue):
        metric = metrics.Metric(scaling, metricValue)
        self.title = metric.title
        self.x_label = x_label
        self.y_label = y_label
        self.data = perf_data(metric)
        self.kind = None

        super().__init__("rectilinear")
        self.colorbar = None

    def update(self):
        if not self.data.metrics.data:
            return
        self.ax.cla()

        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)

        self.ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(log_tick_formatter)
        )
        self.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(log_tick_formatter)
        )

        scatter = self.ax.scatter(
            self.data.xs,
            self.data.ys,
            c=self.data.metrics.data,
            s=3,
            cmap="RdYlGn",
            label=self.kind,
        )

#Add or update the colorbar
        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(
                scatter, ax=self.ax, orientation="vertical"
            )
        else:
            self.colorbar.ax.cla()
            self.colorbar = self.fig.colorbar(
                scatter, cax=self.colorbar.ax, orientation="vertical"
            )

        self.ax.legend()

    def add(self, sample):
        x = sample.primitive[self.x_label]
        y = sample.primitive[self.y_label]
        dt = sample.primitive["dt"]
        if self.kind is None:
            self.kind = sample.kind()
        elif self.kind != sample.kind():
            raise RuntimeError("Only one kind is supported for heatmaps")
        self.data.add(rescale([x, y]), sample)

class heatMap3D(Plot):
    def __init__(self, x_label, y_label, z_label, scaling, metricValue):
        metric = metrics.Metric(scaling, metricValue)
        self.title = metric.title
        self.x_label = x_label
        self.y_label = y_label
        self.z_label = z_label
        self.data = perf_data(metric)
        self.kind = None

        super().__init__()
        self.colorbar = None

    def update(self):
        if not self.data.metrics.data:
            return
        self.ax.cla()

        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.set_zlabel(self.z_label)

        self.ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(log_tick_formatter)
        )
        self.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(log_tick_formatter)
        )
        self.ax.zaxis.set_major_formatter(
            mticker.FuncFormatter(log_tick_formatter)
        )

#n = colors.Normalize(0, 1, clip = True)
        scatter = self.ax.scatter(
            self.data.xs,
            self.data.ys,
            self.data.zs,
            c=self.data.metrics.data,
            s=3,
            cmap="RdYlGn",
            label=self.kind,
        )

#Add the colorbar
        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(
                scatter, ax=self.ax, orientation="vertical"
            )
        else:
            self.colorbar.ax.cla()
            self.colorbar = self.fig.colorbar(
                scatter, cax=self.colorbar.ax, orientation="vertical"
            )

        self.ax.legend()

    def add(self, sample):
        x = sample.primitive[self.x_label]
        y = sample.primitive[self.y_label]
        z = sample.primitive[self.z_label]
        dt = sample.primitive["dt"]
        if self.kind is None:
            self.kind = sample.kind()
        elif self.kind != sample.kind():
            raise RuntimeError("Only one kind is supported for heatmaps")
        self.data.add(rescale([x, y, z]), sample)


class scatter3D(Plot):
    def __init__(self, x_label, y_label, scaling, metricValue):
        self.scaling = scaling
        self.metric_value=metricValue
        self.title = f"{self.scaling.title} {self.metric_value.title}"
        self.x_label = x_label
        self.y_label = y_label
        self.z_label = metricValue.title
        self.data: dict[matmul.Kind, perf_data] = {}

        super().__init__()

    def update(self):
        self.ax.cla()

        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.set_zlabel(self.z_label)

        self.ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(log_tick_formatter)
        )
        self.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(log_tick_formatter)
        )

        for key, value in self.data.items():
            self.ax.scatter(
                value.xs, value.ys, value.metrics.data, s=1, label=key
            )
        self.ax.legend()

    def add(self, sample):
        x = sample.primitive[self.x_label]
        y = sample.primitive[self.y_label]
        dt = sample.primitive["dt"]
        kind = sample.kind()
        if not kind in self.data:
            self.data[kind] = perf_data(metrics.Metric(self.scaling, self.metric_value))
        self.data[kind].add(rescale([x, y]), sample)

def initialize():
    plt.ion()
    plt.draw()

def update():
    plt.draw()
    plt.pause(0.02)

def finalize():
    plt.ioff()
    plt.show()
