################################################################################
# Copyright 2022 Intel Corporation
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
        ofs=','

        def key2str(key, value):
            def mds2str(mds):
                res = ''
                md_fields = ['arg', 'data_type', 'padding', 'format_kind', 'tag']
                ffs=':'
                mdfs=' '
                return mdfs.join([ffs.join([arg[field] for field in md_fields])
                                  for arg in mds])

            if (key == 'mds'):
                return mds2str(value)
            else:
                return str(value)

        #Gather occurences and aggregate time statistics
        total_time = 0
        for key, value in input.items():
            item_key = ofs.join([key2str(k, value[k]) for k in agg_keys])
            occ,time = data.get(item_key, (0, 0.0))
            data[item_key] = (occ + 1, time + float(value['time']))
            total_time += float(value['time'])

        #sort keys by increasing total time
        sorted_item_keys = sorted(data, key=lambda t : data.__getitem__(t)[1],
                                  reverse=True)


        output['all'] = ofs.join(agg_keys + ['ncalls',
                                             'agg_time(ms)',
                                             "overall%"]) + '\n'
        def my_str(s, scale = 1):
            return '{val:.2f}'.format(val=s*scale)
        ors='\n'
        output['all'] += ors.join([ofs.join([str(item_key),
                                             str(data[item_key][0]),
                                             my_str(data[item_key][1]),
                                             my_str(data[item_key][1] / total_time, 100)])
                                   for item_key in sorted_item_keys])
        return output
