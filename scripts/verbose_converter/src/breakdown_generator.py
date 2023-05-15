################################################################################
# Copyright 2022-2023 Intel Corporation
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


class BreakdownGenerator:
    """
    Generates an input for benchdnn from internal representation.
    """

    def __init__(self, writer):
        self.__writer = writer

    def generate(self, input, agg_keys):
        data = {}
        output = {}
        ofs = ","

        def key2str(key, value):
            def mds2str(mds):
                md_fields = [
                    "arg",
                    "data_type",
                    "properties",
                    "format_kind",
                    "tag",
                    "strides",
                ]
                ffs = ":"
                mdfs = " "
                return mdfs.join(
                    [ffs.join([arg[field] for field in md_fields]) for arg in mds]
                )

            def aux2str(aux):
                auxfs = " "
                return auxfs.join([f"{k}:{v}" for k, v in aux.items()])

            if key == "mds":
                return mds2str(value)
            elif key == "aux":
                return aux2str(value)
            else:
                return str(value)

        # Gather occurences and aggregate time statistics
        total_time = 0
        for key, value in input.items():
            item_key = ofs.join([key2str(k, value[k]) for k in agg_keys])
            occ, time = data.get(item_key, (0, 0.0))
            data[item_key] = (occ + 1, time + float(value["time"]))
            total_time += float(value["time"])

        # sort keys by increasing total time
        sorted_item_keys = sorted(
            data, key=lambda t: data.__getitem__(t)[1], reverse=True
        )

        cum_entry = 0
        cum_time = 0
        avg_call = 0
        sorted_avg_call = {}
        sorted_cum_time = {}
        for key in sorted_item_keys:
            cum_entry = cum_entry + 1
            cum_time = cum_time + data[key][1]
            avg_call = avg_call + (data[key][0] - avg_call) / cum_entry
            sorted_avg_call[key] = avg_call
            sorted_cum_time[key] = cum_time

        output["all"] = (
            ofs.join(
                agg_keys
                + [
                    "ncalls",
                    "time(ms)",
                    "overall%",
                    "agg_ncalls(avg)",
                    "agg_time(ms)",
                    "agg_overall%",
                ]
            )
            + "\n"
        )

        def str_num(s):
            return "{val:.2f}".format(val=s)

        def str_pct(s):
            return "{val:.2f}".format(val=s * 100)

        ors = "\n"
        output["all"] += ors.join(
            [
                ofs.join(
                    [
                        str(item_key),
                        str(data[item_key][0]),
                        str_num(data[item_key][1]),
                        str_pct(data[item_key][1] / total_time),
                        str_num(sorted_avg_call[item_key]),
                        str_num(sorted_cum_time[item_key]),
                        str_pct(sorted_cum_time[item_key] / total_time),
                    ]
                )
                for item_key in sorted_item_keys
            ]
        )
        return output
