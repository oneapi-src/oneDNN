################################################################################
# Copyright 2020-2021 Intel Corporation
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


class LogParser:
    """
    Parses a log file with oneDNN verbose and converts it into internal
    representation.
    """
    def __init__(self, writer, input=''):
        # each data entry is a dictionary that consists of:
        # engine(str),
        # primitive(str),
        # implementation(str),
        # prop_kind(str),
        # alg_kind(str),
        # mds({ arg(str) : { data_type(str), format_kind(str), tag(str), flags(str) }})
        # shapes(str)
        # extensions(str)
        # time(float)
        self.__raw_data = []
        self.__data = {}
        self.__writer = writer
        self.__input = input

    def process(self):
        """
        Adds data from the last log file.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        def convert_primitive(log_entry, template):
            """
            Converts oneDNN verbose primitive entry into the internal
            representation.
            """
            def convert_mds(log_mds):
                mds = []
                for md in log_mds.split(' '):
                    # arg_dt:padding:format_kind:tag:flags
                    fields = md.split(':')
                    arg_dt = fields[0]
                    padding = fields[1]
                    format_kind = fields[2]
                    tag = fields[3]
                    flags = {}
                    flags['value'] = fields[4]
                    if len(fields) > 5:
                        flag_fields = fields[5:]
                        for f in flag_fields:
                            if f[:3] == 's8m':
                                flags['s8_comp_mask'] = f[3:]
                            if f[:3] == 'zpm':
                                flags['zp_comp_mask'] = f[3:]

                    data_type = arg_dt.split('_')[-1]
                    arg = arg_dt[:-len(data_type) - 1]
                    mds.append({
                        'arg': arg,
                        'data_type': data_type,
                        'padding': padding,
                        'format_kind': format_kind,
                        'tag': tag,
                        'flags': flags
                    })
                return mds

            def convert_alg(alg):
                found_alg = alg.find('alg')
                if found_alg != -1:
                    alg = alg[len('alg') + 1:]
                return alg

            def convert_prim_kind(prim_kind):
                if prim_kind == 'pooling_v2':
                    prim_kind = 'pooling'
                return prim_kind

            def convert_exts(exts):
                def extract_attr(attrs, type):
                    start_idx = attrs.find(type)
                    if start_idx == -1:
                        return ''

                    start_idx += len(type) + 1
                    end_symbol = ' '
                    end_idx = attrs.find(end_symbol, start_idx)
                    return attrs[start_idx:end_idx]

                def convert_post_ops(value):
                    def convert_binary_post_op(value):
                        fields = value.split(':')
                        alg = fields[0]
                        dt = fields[1]
                        mask = fields[2]
                        tag = None
                        if len(fields) > 3:
                            tag = fields[3]
                        return {
                            'prim_kind': 'binary',
                            'alg': alg,
                            'dt': dt,
                            'mask': mask,
                            'tag': tag
                        }

                    def convert_dw_post_op(value):
                        p_op = {
                            'alg': '',
                            'dst_dt': 'f32',
                            'wei_dt': 'f32',
                            'scales': {
                                'mask': '0',
                                'value': None
                            }
                        }
                        params = value.split(':')
                        len_params = len(params)
                        p_op['alg'] = params[0]
                        if len_params > 1:
                            p_op['dst_dt'] = params[1]
                        if len_params > 2:
                            p_op['wei_dt'] = 's8'
                            p_op['scales']['mask'] = params[2]
                        if len_params > 3:
                            p_op['scales']['value'] = params[3]
                        return p_op

                    def convert_eltwise_post_op(value):
                        p_op = {
                            'alg': '',
                            'alpha': '1.0',
                            'beta': '0.0',
                            'scale': '1.0'
                        }
                        params = value.split(':')
                        len_params = len(params)
                        p_op['alg'] = params[0]
                        if len_params > 1:
                            p_op['alpha'] = params[1]
                        if len_params > 2:
                            p_op['beta'] = params[2]
                        if len_params > 3:
                            p_op['scale'] = params[3]
                        return p_op

                    def convert_sum_post_op(value):
                        p_op = {'alg': '', 'scale': '1.0'}
                        params = value.split(':')
                        len_params = len(params)
                        p_op['alg'] = params[0]
                        if len_params > 1:
                            p_op['scale'] = params[1]
                        return p_op

                    convert = {
                        'binary': convert_binary_post_op,
                        'dw': convert_dw_post_op,
                        'eltwise': convert_eltwise_post_op,
                        'sum': convert_sum_post_op
                    }

                    entries = value.split('+')
                    postops = []
                    for e in entries:
                        for k in convert.keys():
                            if k in e:
                                cvt = convert.get(k)
                                postops.append(cvt(e))
                                break
                    return postops

                def convert_oscale(value):
                    oscale = {'mask': '0', 'value': None}
                    params = value.split(':')
                    len_params = len(params)
                    oscale['mask'] = params[0]
                    if len_params > 1:
                        oscale['value'] = params[1]
                    return oscale

                def convert_scales(value):
                    res = {}
                    scales = value.split('+')
                    for s in scales:
                        scale = {'mask': '0', 'value': None}
                        params = s.split(':')
                        len_params = len(params)
                        arg = params[0]
                        scale['mask'] = params[1]
                        if len_params > 2:
                            scale['value'] = params[2]
                        res[arg] = scale
                    return res

                def convert_zero_points(value):
                    res = {}
                    zero_points = value.split('+')
                    for zp in zero_points:
                        zp_dict = {'mask': '0', 'value': None}
                        params = zp.split(':')
                        arg = params[0]
                        zp_dict['mask'] = params[1]
                        if len(params) > 2:
                            zp_dict['value'] = params[2]
                        res[arg] = zp_dict
                    return res

                def convert_scratchpad_mode(value):
                    return value

                converters = {
                    'attr-post-ops': convert_post_ops,
                    'attr-oscale': convert_oscale,
                    'attr-scales': convert_scales,
                    'attr-zero-points': convert_zero_points,
                    'attr-scratchpad': convert_scratchpad_mode
                }
                attrs = {}
                for e in converters.keys():
                    attr = extract_attr(exts, e)
                    if attr != '':
                        attrs[e] = converters[e](attr)
                return attrs

            def convert_pass(v):
                return v

            convert = {
                'prim_kind': convert_prim_kind,
                'mds': convert_mds,
                'alg_kind': convert_alg,
                'exts': convert_exts
            }

            dnnl_to_ir = {
                'engine': 'engine',
                'prim_kind': 'primitive',
                'impl': 'implementation',
                'prop_kind': 'prop_kind',
                'mds': 'memory_descriptors',
                'exts': 'attributes',
                'alg_kind': 'auxiliary',
                'shapes': 'problem_desc',
                'time': 'exec_time',
                'timestamp': 'timestamp'
            }

            entry = {}

            t = template.split(',')
            for key, value in dnnl_to_ir.items():
                try:
                    idx = t.index(value)
                    if idx != -1:
                        cvt = convert.get(key)
                        if cvt == None:
                            cvt = convert_pass
                        field = log_entry[idx]
                        try:
                            entry[key] = cvt(field)
                        except:
                            self.__writer.print(f"Parser: parsing entry error: {field}: {value}", 'WARN')
                    else:
                        self.__writer.print(f"Parser: Uunknown entry: {value}", 'WARN')
                except:
                    self.__writer.print(f"Parser: skipping empty entry: {key}", 'WARN')
            return entry

        verbose_template = "dnnl_verbose,operation,engine,primitive," + \
            "implementation,prop_kind,memory_descriptors,attributes," + \
            "auxiliary,problem_desc"

        i = len(self.__data)
        for l in self.__input:
            self.__raw_data.append(l.rstrip())
            l_raw = l.split(",")
            marker = l_raw[0]
            if marker == "dnnl_verbose":
                event = l_raw[1]
                if event == "info":
                    opt = l_raw[2]
                    if opt == "prim_template":
                        verbose_template = "dnnl_verbose," + l.split(':')[1]
                if event == "exec":
                    l_converted = convert_primitive(l_raw, verbose_template)
                    if l_converted:
                        self.__data[i] = l_converted
                        i = i + 1

    def get_data(self):
        """
        Returns information about DNN calls.

        Parameters
        ----------
        None

        Returns
        -------
        data
        """

        return self.__data

    def dump(self, converted=False):
        """
        Prints data parsed from input to stdout.

        Parameters
        ----------
        converted (default: False) -- If True dump() prints data in internal
        represenataion, otherwise prints data in the original form.

        Returns
        -------
        None
        """

        if converted:
            [
                self.__writer.print(f"{key}, {value}", 'STDIO')
                for key, value in self.__data.items()
            ]
        else:
            [self.__writer.print(d, 'STDIO') for d in self.__raw_data]
