################################################################################
# Copyright 2024-2025 Intel Corporation
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

import string
from contextlib import nullcontext
from typing import (
    ContextManager,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

from . import ir

__all__ = ["Parser"]


class ParseSpec:
    digits = list(string.digits)

    def __init__(self, buf: str):
        self._buf = buf
        self.offset = 0

    def __str__(self):
        return self.buf

    @property
    def buf(self):
        return self._buf[self.offset :]

    @property
    def eof(self):
        return self.offset >= len(self._buf)

    def peek(self, n=1):
        return self.buf[:n]

    def seek(self, n=1):
        self._read(n)

    def _read(self, n: int) -> str:
        token = self._buf[self.offset : self.offset + n]
        self.offset += n
        return token

    def _find_str(self) -> int:
        buf = ParseSpec(self.buf)
        while not buf.eof and buf.peek() not in ("+", ":"):
            buf.seek()
        return buf.offset

    def _find_uint(self) -> int:
        buf = ParseSpec(self.buf)
        if buf.eof or buf.peek() not in self.digits:
            return 0

        if not buf.read_literal("0"):
            while buf.read_one_of(*self.digits):
                pass
        return buf.offset

    def _find_int(self) -> int:
        buf = ParseSpec(self.buf)
        buf.read_one_of("-", "+")
        return buf.offset + buf._find_uint()

    def _find_float(self) -> int:
        buf = ParseSpec(self.buf)
        buf.read_one_of("-", "+")
        if buf.eof or buf.peek() not in ["."] + self.digits:
            return 0  # ignore [+/-][e...]
        if not buf.read_literal("0"):
            while buf.read_one_of(*self.digits):
                pass
        # else: we already read a 0.
        if buf.read_literal("."):
            while buf.read_one_of(*self.digits):
                pass
        if buf.read_literal("e"):
            buf.read_one_of("-", "+")
            if not buf.read_one_of(*self.digits):
                return 0  # ignore [+/-][X][.Y]e[+/-]
            while buf.read_one_of(*self.digits):
                pass
        return buf.offset

    def _find_literal(self, literal):
        if self.buf.startswith(literal):
            return len(literal)
        return 0

    def read_str(self) -> str:
        return self._read(self._find_str())

    def read_literal(self, literal: str) -> Optional[str]:
        offset = self._find_literal(literal)
        if offset == len(literal):
            return self._read(offset)
        return None

    def read_one_of(self, *literals: str) -> Optional[str]:
        for literal in literals:
            if self.read_literal(literal) is not None:
                return literal
        return None

    def read_uint(self) -> Optional[int]:
        offset = self._find_uint()
        if offset:
            return int(self._read(offset))
        return None

    def read_int(self) -> Optional[int]:
        offset = self._find_int()
        if offset:
            return int(self._read(offset))
        return None

    def read_float(self) -> Optional[float]:
        offset = self._find_float()
        if offset:
            return float(self._read(offset))
        return None


class ParseError(ValueError):
    pass


class InvalidEntryError(ParseError):
    pass


class ParserImpl:
    default_template = (
        "operation,engine,primitive,implementation,prop_kind,"
        + "memory_descriptors,attributes,auxiliary,problem_desc,exec_time"
    )
    _version_map: Dict[int, type] = {}

    @staticmethod
    def parse_aux(aux: str):
        parsed: Dict[str, str] = {}
        if aux == "":
            return parsed
        for aux_l in aux.split():
            # Handle strings like NAME:VAL1[:VAL2[:VAL3...]]
            field, *values = aux_l.split(":", 1)
            parsed[field] = values[0] if values else ""
        return parsed

    def parse_mds(self, descriptors):
        try:
            return list(map(self.parse_md, descriptors.split()))
        except ValueError:
            raise ValueError(f"Could not parse mds {descriptors}")

    @staticmethod
    def is_bit_layout(dt):
        buf = ParseSpec(dt)
        if not buf.read_literal("e"):
            return False
        if buf.read_uint() is None:
            return False
        if not buf.read_literal("m"):
            return False
        if buf.read_uint() is None:
            return False
        return buf.eof  # eXmY

    def is_float_type(self, dt):
        buf = ParseSpec(dt)
        buf.read_literal("b")  # ignore b in bf16
        if not buf.read_literal("f"):
            return False
        if buf.read_uint() is None:
            return False
        if buf.eof:
            return True  # bf16, f16, f32, f64
        if not buf.read_literal("_"):
            return False
        return self.is_bit_layout(buf.buf)  # fZ_eXmY

    @staticmethod
    def is_int_type(dt):
        buf = ParseSpec(dt)
        if not buf.read_one_of("u", "s"):
            return False
        if buf.read_uint() is None:
            return False
        return buf.eof

    def is_data_type(self, dt):
        return (
            dt == "undef"
            or self.is_int_type(dt)
            or self.is_float_type(dt)
            or self.is_bit_layout(dt)
        )

    @staticmethod
    def parse_md_flags(flags, fields):
        flags = ir.MemoryDescriptor.Flags(value=flags or "f0")
        for field in fields:
            if field[:3] == "s8m":
                flags.s8_comp_mask = field[3:]
            elif field[:3] == "zpm":
                flags.zp_comp_mask = field[3:]
            elif field[:2] == "sa":
                flags.scale_adjust = float(field[2:])
        return flags

    def parse_md(self, descriptor):
        fields = descriptor.split(":")
        arg_dt, properties, format_kind, tag = fields[:4]
        arg_dt_parts = arg_dt.split("_")
        for i in range(1, len(arg_dt_parts)):
            arg = "_".join(arg_dt_parts[:i])
            dt = "_".join(arg_dt_parts[i:])
            if self.is_data_type(dt):
                break
        else:
            if len(arg_dt_parts) != 1 or not self.is_data_type(arg_dt):
                raise ParseError(
                    f"Could not parse memory descriptor {descriptor}"
                )
            arg, dt = "data", arg_dt

        strides = ""
        if "f" not in fields[4] and format_kind != "undef":
            strides = fields[4]
            flags = self.parse_md_flags(fields[5], fields[6:])
        else:
            flags = self.parse_md_flags(fields[4], fields[5:])
        return ir.MemoryDescriptor(
            arg=arg,
            data_type=dt,
            properties=properties,
            format_kind=format_kind,
            tag=tag,
            strides=strides,
            flags=flags,
        )

    def parse_attrs(self, attrs):
        parsed = {}
        for attr in attrs.split():
            spec = ParseSpec(attr)
            name, args = spec.read_str(), ""
            if spec.read_literal(":"):
                args = spec.buf
            if name == "attr-acc-mode":
                parsed[name] = self.parse_acc_mode(args)
            elif name == "attr-deterministic":
                parsed[name] = self.parse_deterministic(args)
            elif name == "attr-dropout":
                parsed[name] = self.parse_dropout(args)
            elif name == "attr-fpmath":
                parsed[name] = self.parse_fpmath_mode(args)
            # Kept for compatibility with v2.7 and below.
            elif name == "attr-oscale":
                parsed[name] = self.parse_oscale(args)
            elif name == "attr-post-ops":
                parsed[name] = self.parse_post_ops(args)
            elif name == "attr-rounding-mode":
                parsed[name] = self.parse_rounding_modes(args)
            elif name == "attr-scales":
                parsed[name] = self.parse_scales(args)
            elif name == "attr-scratchpad":
                parsed[name] = self.parse_scratchpad_mode(args)
            elif name == "attr-zero-points":
                parsed[name] = self.parse_zero_points(args)
        return ir.Attributes(parsed)

    def parse_post_ops(self, post_ops: str):
        spec = ParseSpec(post_ops)
        parsed: List[ir.PostOp] = []
        while True:
            alg = spec.read_str()
            if alg == "sum":
                parsed.append(self.parse_sum_post_op(spec))
            elif alg == "dw":
                parsed.append(self.parse_dw_post_op(spec))
            elif alg == "prelu":
                parsed.append(self.parse_prelu_post_op(spec))
            elif alg.startswith("eltwise_"):
                parsed.append(self.parse_eltwise_post_op(spec, alg))
            elif alg.startswith("binary_"):
                parsed.append(self.parse_binary_post_op(spec, alg))
            else:
                raise ParseError(f"Unexpected post-op: {alg}")
            if not spec.read_literal("+"):
                break
        return parsed

    @staticmethod
    def parse_sum_post_op(spec) -> ir.SumPostOp:
        post_op = ir.SumPostOp()
        if spec.read_literal(":"):
            post_op.scale = spec.read_float()
        if spec.read_literal(":"):
            post_op.zp = spec.read_int()
        if spec.read_literal(":"):
            post_op.dt = spec.read_str()
        return post_op

    @staticmethod
    def parse_dw_post_op(spec) -> ir.DepthwisePostOp:
        if not spec.read_literal(":"):
            raise ParseError("Expected argument for depthwise post-op")
        ksp = spec.read_str()
        post_op = ir.DepthwisePostOp(ksp=ksp)
        if spec.read_literal(":"):
            post_op.dst_dt = spec.read_str()
        if spec.read_literal(":"):
            post_op.wei_dt = "s8"
            post_op.scales.mask = spec.read_uint()
        if spec.read_literal(":"):
            post_op.scales.value = spec.read_str()
        return post_op

    @staticmethod
    def parse_prelu_post_op(spec) -> ir.PreLUPostOp:
        post_op = ir.PreLUPostOp()
        if spec.read_literal(":"):
            post_op.mask = spec.read_uint()
        if spec.read_literal(":"):
            post_op.has_scaleshift = spec.read_str() == "true"
        return post_op

    @staticmethod
    def parse_eltwise_post_op(spec, alg) -> ir.EltwisePostOp:
        post_op = ir.EltwisePostOp(alg=alg)
        if spec.read_literal(":"):
            post_op.alpha = spec.read_float()
        if spec.read_literal(":"):
            post_op.beta = spec.read_float()
        if spec.read_literal(":"):
            post_op.scale = spec.read_float()
        return post_op

    @staticmethod
    def parse_binary_post_op(spec, alg) -> ir.BinaryPostOp:
        if not spec.read_literal(":"):
            raise ParseError("Expected data type for binary post-op")
        dt = spec.read_str()
        post_op = ir.BinaryPostOp(alg=alg, dt=dt)
        if spec.read_literal(":"):
            post_op.mask = spec.read_uint()
        if spec.read_literal(":"):
            post_op.tag = spec.read_str()
        return post_op

    @staticmethod
    def parse_dropout(args: str) -> ir.Dropout:
        return ir.Dropout(tag=args if args else None)

    @staticmethod
    def parse_per_argument(attr, name, parse):
        spec = ParseSpec(attr)
        parsed = {}
        while True:
            arg = spec.read_str()
            if not spec.read_literal(":"):
                raise ParseError(f"Expected mask for {arg} {name}")
            parsed[arg] = parse(spec)
            if not spec.read_literal("+"):
                break
        return parsed

    def parse_scales(self, scales: str):
        return self.parse_per_argument(scales, "scale", self.parse_scale)

    @staticmethod
    def parse_quantization_param(spec, read_value, param_type):
        # Old style: mask[:[value[*]|*]]
        # New style: mask[:data_type[:groups]]
        param = param_type()
        param.mask = spec.read_uint()
        if spec.read_literal(":"):
            value = read_value()
            if value is not None:
                param.value = value
                spec.read_literal("*")
            elif spec.read_literal("*"):
                pass
            elif not spec.eof:  # new style
                param.data_type = spec.read_str()
                if spec.read_literal(":"):
                    param.groups = spec.read_str()
        return param

    # v2.7 and below
    def parse_oscale(self, oscale: str):
        spec = ParseSpec(oscale)
        return self.parse_scale(spec)

    def parse_scale(self, spec) -> ir.Scale:
        return self.parse_quantization_param(spec, spec.read_float, ir.Scale)

    def parse_zero_points(self, zps: str):
        return self.parse_per_argument(zps, "zero point", self.parse_zero_point)

    def parse_zero_point(self, spec) -> ir.ZeroPoint:
        return self.parse_quantization_param(spec, spec.read_int, ir.ZeroPoint)

    @staticmethod
    def parse_fpmath_mode(mathmode: str) -> ir.FPMathMode:
        spec = ParseSpec(mathmode)
        mode = spec.read_str()
        apply_to_int = False
        if spec.read_literal(":"):
            apply_to_int = spec.read_str() == "true"
        return ir.FPMathMode(mode=mode, apply_to_int=apply_to_int)

    @staticmethod
    def parse_rounding_mode(rounding_mode: str) -> ir.RoundingMode:
        rm = rounding_mode.lower()
        for member in ir.RoundingMode.__members__.values():
            if str(member) == rm:
                return member
        else:
            raise ValueError(f"Invalid rounding mode {rounding_mode}")

    def parse_rounding_modes(self, rounding_modes: str):
        spec = ParseSpec(rounding_modes)
        modes: Dict[str, ir.RoundingMode] = {}
        while True:
            arg = spec.read_str()
            if not spec.read_literal(":"):
                raise ValueError("Expected rounding mode")
            mode = self.parse_rounding_mode(spec.read_str())
            modes[arg] = mode
            if not spec.read_literal("+"):
                break
        return modes

    identity = staticmethod(lambda x: x)

    # Additional attributes
    parse_acc_mode = identity
    parse_deterministic = identity
    parse_scratchpad_mode = identity

    # Additional template components
    parse_operation = identity
    parse_prim_kind = identity
    parse_prop_kind = identity
    parse_engine = identity
    parse_impl = identity
    parse_shapes = identity
    parse_time = staticmethod(float)
    parse_timestamp = staticmethod(float)

    def dnnl_to_ir(self):
        return {
            "operation": ("operation", self.parse_operation, True),
            "engine": ("engine", self.parse_engine, True),
            "primitive": ("prim_kind", self.parse_prim_kind, True),
            "implementation": ("impl", self.parse_impl, True),
            "prop_kind": ("prop_kind", self.parse_prop_kind, True),
            "memory_descriptors": ("mds", self.parse_mds, True),
            "attributes": ("exts", self.parse_attrs, True),
            "auxiliary": ("aux", self.parse_aux, True),
            "problem_desc": ("shapes", self.parse_shapes, True),
            "exec_time": ("time", self.parse_time, False),
            "timestamp": ("timestamp", self.parse_timestamp, False),
        }

    def parse(self, line: str, template: Optional[str]):
        if template is None:
            template = self.default_template
        entry = {}
        fields = template.rstrip().split(",")
        values = line.rstrip().split(",")
        mapping = self.dnnl_to_ir()
        min_fields = sum((mapping[field][2] for field in fields))
        max_fields = len(fields)
        if len(values) < min_fields:
            raise InvalidEntryError("parse error: too few fields to parse")
        if len(values) > max_fields:
            raise InvalidEntryError("parse error: too many fields to parse")
        mapped = dict(zip(fields, values))
        for field, (key, parse, reqd) in mapping.items():
            if field not in mapped:
                if not reqd:
                    continue
                raise InvalidEntryError(f"parse error: missing {field} field")
            value = mapped[field]
            try:
                entry[key] = parse(value)
            except (ParseError, ValueError) as e:
                raise ParseError(f"parse error: {field}: {value} ({e!s})")
        return entry


def register(*, version: int):
    def registrar(impl: type):
        ParserImpl._version_map[version] = impl
        return impl

    return registrar


@register(version=0)
class LegacyParserImpl(ParserImpl):
    pass


@register(version=1)
class V1ParserImpl(ParserImpl):
    def parse_md(self, descriptor):
        fields = descriptor.split(":")
        return ir.MemoryDescriptor(
            arg=fields[0],
            data_type=fields[1],
            properties=fields[2],
            format_kind=fields[3],
            tag=fields[4],
            strides=fields[5],
            flags=self.parse_md_flags(fields[6], fields[7:]),
        )


class Parser:
    _parser_impls: Dict[int, ParserImpl] = {}
    _default_events = "exec", "create", "create_nested"

    def __init__(
        self,
        input: Iterable[str],
        events: Iterable[str] = _default_events,
        error_handler: ContextManager = nullcontext(),
    ):
        self.input = input
        self.events = set(events)
        self.error_handler = error_handler

    def _fix_template(self, template) -> Optional[str]:
        return template

    @staticmethod
    def _parse_leading_fields(input: Iterable[str]):
        MARKER = "onednn_verbose"
        for line in map(str.rstrip, input):
            if not line.startswith(f"{MARKER},"):
                continue
            try:
                _, operation, args = line.split(",", 2)
            except ValueError:
                continue
            version = 0
            if operation.startswith("v"):
                try:
                    version = int(operation[1:])
                except ValueError:
                    pass
                else:
                    operation, args = args.split(",", 1)
            timestamp = None
            try:
                timestamp = float(operation)
            except ValueError:
                pass
            else:
                operation, args = args.split(",", 1)
            component = "primitive"
            if operation in ("graph", "primitive", "ukernel"):
                component = operation
                operation, args = args.split(",", 1)
            yield line, version, timestamp, component, operation, args

    def __iter__(self) -> Iterator[Tuple[str, ir.Entry]]:
        template = None
        cache: Dict[str, dict] = {}
        errors: Set[str] = set()
        parsed = self._parse_leading_fields(self.input)
        for line, version, timestamp, component, operation, args in parsed:
            if component == "graph":
                continue
            event = operation.split(":", 1)[0]
            if event == "info":
                for marker in ("template", "prim_template"):
                    if not args.startswith(f"{marker}:"):
                        continue
                    fixed_template = self._fix_template(args[len(marker) + 1 :])
                    if fixed_template is not None:
                        break
                else:
                    continue
                first_component, rest = fixed_template.split(",", 1)
                # Timestamp is usually out of order with respect to the
                # template because of missing component for "graph",
                # "primitive", "ukernel", etc.
                if first_component == "timestamp":
                    fixed_template = rest
                if template != fixed_template:
                    template = fixed_template
                    cache.clear()
                continue
            if event not in self.events:
                continue
            leading_args, last_arg = args.rsplit(",", 1)
            try:
                time = float(last_arg)
            except ValueError:
                time = 0.0
                leading_args = args
            key = f"v{version},{component},{operation},{leading_args}"
            if key in errors:
                continue
            success = False
            with self.error_handler:
                if key in cache:
                    params = dict(cache[key])
                    params.update(time=time, timestamp=timestamp)
                else:
                    new_line = f"{operation},{args}"
                    params = self.parse(new_line, template, version)
                    cache[key] = dict(params)
                    if timestamp is not None:
                        params.update(timestamp=timestamp)
                yield line, ir.Entry(version=version, **params)
                success = True
            if not success:
                errors.add(key)

    def items(self) -> Iterable[Tuple[int, Tuple[str, ir.Entry]]]:
        yield from enumerate(self)

    @staticmethod
    def _get_impl(version: int = 0) -> ParserImpl:
        if version not in Parser._parser_impls:
            if version not in ParserImpl._version_map:
                raise ParseError(f"No parsers registered for version {version}")
            Parser._parser_impls[version] = ParserImpl._version_map[version]()
        return Parser._parser_impls[version]

    def parse(self, line: str, template: Optional[str], version: int = 0):
        impl = self._get_impl(version)
        return impl.parse(line, template)
