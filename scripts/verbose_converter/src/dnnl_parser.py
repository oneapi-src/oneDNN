################################################################################
# Copyright 2020-2024 Intel Corporation
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

    def __init__(self, writer, input=""):
        # each data entry is a dictionary that consists of:
        # engine(str),
        # primitive(str),
        # implementation(str),
        # prop_kind(str),
        # aux({field(str) : value(str)}),
        # mds(
        #     {
        #         arg(str): {
        #             data_type(str),
        #             properties(str),
        #             format_kind(str),
        #             tag(str),
        #             strides(str),
        #             flags(str),
        #         }
        #     }
        # )
        # shapes(str)
        # extensions(str)
        # time(float)
        self.__raw_data = []
        self.__data = {}
        self.__writer = writer
        self.__input = input

    def process(self, filter_events):
        """
        Adds data from the last log file.

        Parameters:
        -----------
        filter_events -- List of events to parse, other events are ignored.

        Returns:
        --------
        None
        """

        def convert_primitive(log_entry, template, version):
            """
            Converts oneDNN verbose primitive entry into the internal
            representation.
            """

            def split_arg_dt(arg_dt):
                def buffer(dt):
                    return {"data": dt, "offset": 0}

                def eof(buf):
                    return buf["offset"] >= len(buf["data"])

                def get_data(buf):
                    if eof(buf):
                        return None
                    return buf["data"][buf["offset"] :]

                def read_int(buf):
                    data = get_data(buf)
                    if not data:
                        return None
                    if data[0] not in "123456789":
                        return None
                    for n, c in enumerate(data):
                        if c not in "0123456789":
                            buf["offset"] += n
                            return int(data[:n])
                    buf["offset"] += len(data)
                    return int(data)

                def read_literal(buf, literal):
                    data = get_data(buf)
                    if not data:
                        return None
                    if not data.startswith(literal):
                        return None
                    buf["offset"] += len(literal)
                    return True

                def parse_int_type(dt):
                    buf = buffer(dt)
                    if not (read_literal(buf, "u") or read_literal(buf, "s")):
                        return False
                    if not read_int(buf):
                        return False
                    return eof(buf)

                def parse_float_type(dt):
                    buf = buffer(dt)
                    read_literal(buf, "b")  # ignore b in bf16
                    if not read_literal(buf, "f"):
                        return False
                    if not read_int(buf):
                        return False
                    if eof(buf):
                        return True  # f16, f32, f64
                    if not read_literal(buf, "_e"):
                        return False
                    if not read_int(buf):
                        return False
                    if not read_literal(buf, "m"):
                        return False
                    if not read_int(buf):
                        return False
                    return eof(buf)  # f8_eXmY

                parts = arg_dt.split("_")
                for split in range(1, len(parts)):
                    input_parts = parts[:split]
                    dt_parts = parts[split:]
                    dt = "_".join(dt_parts)
                    if dt == "undef":
                        return "_".join(input_parts), dt
                    if parse_int_type(dt) or parse_float_type(dt):
                        return "_".join(input_parts), dt

            def convert_mds(log_mds, version):
                mds = []
                for md in log_mds.split(" "):
                    fields = md.split(":")
                    idx = 0

                    # if version >= 1:
                    #     arg:dt:properties:format_kind:tag:strides:flags
                    ##       ^
                    # else:
                    #     arg_dt:properties:format_kind:tag:strides:flags
                    # (note) Legacy way could have collisions with `arg` and
                    #   `dt` since `_` used as a delimiter and as a part of the
                    #   name.
                    arg = None
                    data_type = None
                    if int(version) >= 1:
                        arg = fields[idx]
                        idx += 1
                        data_type = fields[idx]
                        idx += 1
                    else:
                        arg_dt = fields[idx]
                        idx += 1
                        arg, data_type = split_arg_dt(arg_dt)

                    properties = fields[idx]
                    idx += 1
                    format_kind = fields[idx]
                    idx += 1
                    tag = fields[idx]
                    idx += 1

                    # Add compatibility for v3.1 verbose and below,
                    # when strides delimeter is absent.
                    # TODO: remove eventually.
                    strides = ""
                    if "f" not in fields[idx] and format_kind != "undef":
                        strides = fields[idx]
                        idx += 1

                    flags = {}
                    flags["value"] = fields[idx]
                    idx += 1
                    if len(fields) > idx:
                        flag_fields = fields[idx:]
                        for f in flag_fields:
                            if f[:3] == "s8m":
                                flags["s8_comp_mask"] = f[3:]
                            if f[:3] == "zpm":
                                flags["zp_comp_mask"] = f[3:]

                    mds.append(
                        {
                            "arg": arg,
                            "data_type": data_type,
                            "properties": properties,
                            "format_kind": format_kind,
                            "tag": tag,
                            "strides": strides,
                            "flags": flags,
                        }
                    )
                return mds

            def convert_aux(log_aux, version):
                aux = {}
                if log_aux == "":
                    return aux
                for log_aux_l in log_aux.split(" "):
                    # Handle strings like NAME:VAL1[:VAL2[:VAL3...]]
                    res = log_aux_l.split(":")
                    field = res[0]
                    value = ""
                    last_idx = len(res) - 1
                    for i in range(1, last_idx):
                        val_i = res[i]
                        value += f"{val_i}:"
                    val_n = res[last_idx]
                    value += f"{val_n}"
                    aux[field] = value
                return aux

            def convert_prim_kind(prim_kind, version):
                return prim_kind

            def convert_exts(exts, version):
                def extract_attr(attrs, type):
                    start_idx = attrs.find(type)
                    if start_idx == -1:
                        return ""

                    start_idx += len(type) + 1
                    end_symbol = " "
                    end_idx = attrs.find(end_symbol, start_idx)
                    if end_idx == -1:
                        end_idx = None
                    return attrs[start_idx:end_idx]

                def convert_structure_to_ir_seq(ir, value):
                    params = value.split(":")
                    fields = list(ir.keys())
                    ir.update(
                        (fields[i], params[i])
                        for i in range(0, min(len(params), len(fields)))
                    )
                    return ir

                def convert_post_ops(value):
                    def convert_binary_post_op(value):
                        p_op = {"alg": "", "dt": "f32", "mask": "0", "tag": None}
                        p_op = convert_structure_to_ir_seq(p_op, value)
                        p_op["prim_kind"] = "binary"
                        return p_op

                    def convert_dw_post_op(value):
                        p_op = {
                            "alg": "",
                            "ksp": "",
                            "dst_dt": "f32",
                            "wei_dt": "f32",
                            "scales": {"mask": "0", "value": None},
                        }
                        params = value.split(":")
                        len_params = len(params)
                        p_op["alg"] = params[0]
                        p_op["ksp"] = params[1]
                        if len_params > 2:
                            p_op["dst_dt"] = params[2]
                        if len_params > 3:
                            p_op["wei_dt"] = "s8"
                            p_op["scales"]["mask"] = params[3]
                        if len_params > 4:
                            p_op["scales"]["value"] = params[4]
                        return p_op

                    def convert_eltwise_post_op(value):
                        p_op = {
                            "alg": "",
                            "alpha": "1.0",
                            "beta": "0.0",
                            "scale": "1.0",
                        }
                        return convert_structure_to_ir_seq(p_op, value)

                    def convert_sum_post_op(value):
                        p_op = {"alg": "", "scale": "1.0", "zp": "0", "dt": ""}
                        return convert_structure_to_ir_seq(p_op, value)

                    def convert_prelu_post_op(value):
                        p_op = {"alg": "", "mask": "0"}
                        return convert_structure_to_ir_seq(p_op, value)

                    convert = {
                        "binary": convert_binary_post_op,
                        "dw": convert_dw_post_op,
                        "eltwise": convert_eltwise_post_op,
                        "sum": convert_sum_post_op,
                        "prelu": convert_prelu_post_op,
                    }

                    entries = value.split("+")
                    postops = []
                    for e in entries:
                        for k in convert.keys():
                            if k in e:
                                cvt = convert.get(k)
                                postops.append(cvt(e))
                                break
                    return postops

                def convert_scales(value):
                    res = {}
                    scales = value.split("+")
                    for s in scales:
                        arg = s[: s.find(":")]
                        s_wo_arg = s[s.find(":") + 1 :]
                        scale_dict = {"mask": "0", "data_type": "f32", "groups": ""}
                        res[arg] = convert_structure_to_ir_seq(scale_dict, s_wo_arg)
                    return res

                def convert_zero_points(value):
                    res = {}
                    zp_value = value.split("+")
                    for zp in zp_value:
                        arg = zp[: zp.find(":")]
                        zp_value_wo_arg = zp[zp.find(":") + 1 :]
                        zp_dict = {"mask": "0", "data_type": "s32", "groups": ""}
                        res[arg] = convert_structure_to_ir_seq(zp_dict, zp_value_wo_arg)
                    return res

                def convert_rounding_mode(value):
                    res = {}
                    rounding_modes = value.split("+")
                    for r in rounding_modes:
                        arg = r[: r.find(":")]
                        res[arg] = r[r.find(":") + 1 :]
                    return res

                def convert_scratchpad_mode(value):
                    return value

                def convert_fpmath_mode(value):
                    return value

                def convert_acc_mode(value):
                    return value

                def convert_dropout(value):
                    res = {}
                    elems = value.split(":")
                    if len(elems) > 0:
                        res["tag"] = elems[0]
                    return res

                def convert_deterministic(value):
                    return value

                converters = {
                    "attr-post-ops": convert_post_ops,
                    "attr-scales": convert_scales,
                    "attr-zero-points": convert_zero_points,
                    "attr-scratchpad": convert_scratchpad_mode,
                    "attr-fpmath": convert_fpmath_mode,
                    "attr-acc": convert_acc_mode,
                    "attr-rounding-mode": convert_rounding_mode,
                    "attr-dropout": convert_dropout,
                    "attr-deterministic": convert_deterministic,
                }
                attrs = {}
                for e in converters.keys():
                    attr = extract_attr(exts, e)
                    if attr != "":
                        attrs[e] = converters[e](attr)
                return attrs

            def convert_pass(v, version):
                return v

            convert = {
                "prim_kind": convert_prim_kind,
                "mds": convert_mds,
                "aux": convert_aux,
                "exts": convert_exts,
            }

            dnnl_to_ir = {
                "engine": "engine",
                "prim_kind": "primitive",
                "impl": "implementation",
                "prop_kind": "prop_kind",
                "mds": "memory_descriptors",
                "exts": "attributes",
                "aux": "auxiliary",
                "shapes": "problem_desc",
                "time": "exec_time",
                "timestamp": "timestamp",
            }

            ir_req = [
                "engine",
                "prim_kind",
                "impl",
                "prop_kind",
                "mds",
                "exts",
                "aux",
                "shapes",
            ]

            entry = {}

            t = template.split(",")
            for key, value in dnnl_to_ir.items():
                notification_level = "WARN" if key in ir_req else "INFO"
                try:
                    idx = t.index(value)
                    if idx != -1:
                        cvt = convert.get(key)
                        if cvt is None:
                            cvt = convert_pass
                        field = log_entry[idx]
                        try:
                            entry[key] = cvt(field, version)
                        except:
                            self.__writer.print(
                                f"Parser: parsing entry error: {field}: {value}",
                                notification_level,
                            )
                    else:
                        self.__writer.print(
                            f"Parser: Unknown entry: {value}", notification_level
                        )
                except:
                    self.__writer.print(
                        f"Parser: skipping empty entry: {key}", notification_level
                    )
            return entry

        # `verbose_template` should have `component` field as second entry, but
        # since it gets discarded for compatibility with previous verbose
        # outputs, it's not in the final version of the string.
        # Restore `component` when the least compatible library version's
        # verbose output will contain it.
        verbose_template = (
            "onednn_verbose,operation,engine,primitive,"
            + "implementation,prop_kind,memory_descriptors,attributes,"
            + "auxiliary,problem_desc"
        )

        i = len(self.__data)
        for line in self.__input:
            self.__raw_data.append(line.rstrip())
            l_raw = line.split(",")
            marker = l_raw[0]
            if marker != "onednn_verbose":
                continue

            verbose_version = 0
            # Check for version presence, discard 'v' from numerical version,
            # and discard version entry for compatibility reasons.
            # Note: to compare against `version`, one must use int() function
            # call as arg is passed as `str` object!
            if l_raw[1][0] == "v" and l_raw[1][1].isdigit():
                verbose_version = l_raw[1].lstrip("v")
                l_raw.pop(1)

            # Discard a timestamp when it's supplied in a standalone line.
            # TODO: update verbose_template instead.
            if l_raw[1].split(".")[0].isdigit():
                l_raw.pop(1)
            # Skip Graph component as not supported
            if l_raw[1] == "graph":
                continue
            # Remove a component from the line if presented (see a comment above)
            if l_raw[1] == "primitive" or l_raw[1] == "ukernel":
                l_raw.pop(1)

            event = l_raw[1].split(":")[0]
            if event == "info":
                opt = l_raw[2]
                if opt.split(":")[0] == "template":
                    verbose_template = "onednn_verbose," + line.split(":")[1]
            if event in filter_events:
                l_converted = convert_primitive(
                    l_raw, verbose_template + ",exec_time", verbose_version
                )
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
                self.__writer.print(f"{key}, {value}", "STDIO")
                for key, value in self.__data.items()
            ]
        else:
            [self.__writer.print(d, "STDIO") for d in self.__raw_data]
