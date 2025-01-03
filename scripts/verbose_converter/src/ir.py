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

import enum
import string
from abc import abstractmethod
from collections.abc import MutableMapping
from dataclasses import MISSING, dataclass, fields
from typing import Dict, List, Optional, Union


def alias(attr):
    def getter(self):
        return getattr(self, attr)

    def setter(self, value):
        return setattr(self, attr, value)

    def deleter(self):
        return delattr(self, attr)

    return property(getter, setter, deleter, attr)


def hash_str(obj):
    return getattr(obj.__class__, "__hash_str__", str)(obj)


@dataclass(eq=False)
class Mapping(MutableMapping):
    def __getitem__(self, item):
        try:
            value = getattr(self, item)
            if isinstance(value, int):
                value = str(value)
            elif isinstance(value, float):
                value = str(value)
                # The verbose converter assumes defaults are 1.0, whereas
                # oneDNN assumes defaults are 0.0. This is a workaround so that
                # we don't accidentally drop these values, instead setting as 0
                # or 1 which will always be sent through to the benchdnn
                # reproducer
                if value[-2:] == ".0":
                    value = value[:-2]
            return value
        except AttributeError:
            raise KeyError(item)

    def __setitem__(self, item, value):
        setattr(self, item, value)

    def __delitem__(self, item):
        delattr(self, item)

    def __len__(self):
        return len(fields(self))

    def __iter__(self):
        for field in fields(self):
            yield field.name

    def __hash__(self):
        return hash(hash_str(self))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return hash_str(self) == hash_str(other)

    def __str__(self):
        raise NotImplementedError

    def __hash_str__(self):
        return str(self)

    def __repr__(self):
        child_reprs = []
        for key, value in self.items():
            child_reprs.append(f"{key!r}: {value!r}")
        return "{" + ", ".join(child_reprs) + "}"


@dataclass(eq=False)
class MemoryDescriptor(Mapping):
    @dataclass(eq=False)
    class Flags(Mapping):
        value: str
        s8_comp_mask: Optional[str] = None
        zp_comp_mask: Optional[str] = None
        scale_adjust: float = 1.0

        def __str__(self):
            my_str = self.value
            if self.s8_comp_mask is not None:
                my_str += f":s8m{self.s8_comp_mask}"
            if self.zp_comp_mask is not None:
                my_str += f":s8m{self.zp_comp_mask}"
            if self.scale_adjust != 1.0:
                my_str += f":sa{self.scale_adjust}"
            return my_str

    arg: str
    data_type: str
    properties: str
    format_kind: str
    tag: str
    flags: Flags
    strides: str = ""  # Pre-v3.1 does not have strides

    padding = alias("properties")

    def __len__(self):
        return 1 + super().__len__()

    def __iter__(self):
        yield from super().__iter__()
        yield "padding"

    def _format(self, tag: str, convert) -> str:
        header = f"{self.arg}:{self.data_type}"
        return ":".join(
            [
                header,
                self.properties,
                self.format_kind,
                tag,
                self.strides,
                convert(self.flags),
            ]
        )

    def __str__(self):
        return self._format(self.tag, str)

    def __hash_str__(self):
        tag = self.tag
        if "a" not in self.properties:
            return self._format(tag, hash_str)
        for i, c in enumerate(tag):
            if not c.isalpha():
                return self._format(string.ascii_lowercase[:i], hash_str)
        return self._format(string.ascii_lowercase[: len(tag)], hash_str)


@dataclass(eq=False)
class Dropout(Mapping):
    tag: Optional[str] = None

    def __str__(self):
        return self.tag or ""


class FormattedMapping(Mapping):
    @abstractmethod
    def _format(self, _) -> str:
        raise NotImplementedError

    def __str__(self):
        return self._format(str)

    def __hash_str__(self):
        return self._format(hash_str)


@dataclass(eq=False)
class PostOp(FormattedMapping):
    alg: str

    def _format(self, convert):
        required_args = []
        optional_args = []
        seen_non_default = False
        for field in reversed(fields(self)):
            if field.name == "alg":
                continue
            value = getattr(self, field.name)
            if field.default is MISSING:
                required_args.append(value)
                continue
            if not seen_non_default and value == field.default:
                continue
            seen_non_default = True
            optional_args.append(value)
        args = [self.alg] + required_args[::-1] + optional_args[::-1]
        return ":".join(map(convert, args))


@dataclass(eq=False)
class SumPostOp(PostOp):
    alg: str = "sum"
    scale: float = 1.0
    zp: int = 0
    dt: str = ""


@dataclass(eq=False)
class DepthwiseScales(Mapping):
    mask: int = 0
    value: Optional[str] = None

    def __str__(self):
        if self.value is not None:
            return f"{self.mask}:{self.value}"
        if self.mask != 0:
            return str(self.mask)
        return ""


@dataclass(eq=False)
class KSPMixin:
    ksp: str


@dataclass(eq=False)
class DepthwisePostOp(PostOp, KSPMixin):
    alg: str = "dw"
    dst_dt: str = "f32"
    wei_dt: str = "f32"
    scales: DepthwiseScales = DepthwiseScales()

    def __len__(self):
        return 1 + super().__len__()

    def __iter__(self):
        yield "alg"
        yield from super().__iter__()


@dataclass(eq=False)
class PreLUPostOp(PostOp):
    alg: str = "prelu"
    mask: int = 0
    has_scaleshift: bool = False

    def __getitem__(self, item):
        if item == "has_scaleshift":
            return "true" if self.has_scaleshift else ""
        return super().__getitem__(item)

    def __str__(self):
        if self.has_scaleshift:
            return f"{self.alg}:{self.mask}:true"
        return f"{self.alg}:{self.mask}"


@dataclass(eq=False)
class EltwisePostOp(PostOp):
    alpha: float = 0.0
    beta: float = 0.0
    scale: float = 1.0


@dataclass(eq=False)
class BinaryPostOp(PostOp):
    dt: str
    mask: int = 0
    tag: str = "abx"


@dataclass(eq=False)
class QuantizationParam(Mapping):
    value: float
    data_type: str
    mask: int = 0
    groups: str = ""

    def __str__(self):
        if self.groups is not None:
            return f"{self.mask}:{self.data_type}:{self.groups}"
        return f"{self.mask}:{self.data_type}"


@dataclass(eq=False)
class Scale(QuantizationParam):
    value: float = 1.0
    data_type: str = "f32"


@dataclass(eq=False)
class ZeroPoint(QuantizationParam):
    value: int = 0
    data_type: str = "s32"


class CompositeAttribute:
    def __str__(self):
        raise NotImplementedError


@dataclass(eq=False)
class FPMathMode(CompositeAttribute):
    mode: str
    apply_to_int: bool = False

    def __str__(self):
        a2i_str = ":true" if self.apply_to_int else ""
        return self.mode + a2i_str


class RoundingMode(CompositeAttribute, enum.Enum):
    ENVIRONMENT = "environment"
    STOCHASTIC = "stochastic"

    def __str__(self):
        return self.value


Attribute = Union[
    str,  # acc-mode, etc
    FPMathMode,
    Dropout,
    List[PostOp],
    Dict[str, Scale],
    Dict[str, ZeroPoint],
    Dict[str, RoundingMode],
    Scale,  # oscale
]


def attribute_accessor(name):
    name = "attr-" + name

    def getter(self) -> Attribute:
        return self._attributes.get(name)

    def setter(self, value):
        self._attributes[name] = value

    def deleter(self):
        if name in self._attributes:
            del self._attributes[name]

    return property(getter, setter, deleter)


@dataclass(eq=False, repr=False)
class Attributes(Mapping):
    def __init__(self, attributes: Optional[Dict[str, Attribute]] = None):
        if attributes is None:
            attributes = {}
        self._attributes: Dict[str, Attribute] = attributes

    def __getitem__(self, item: str):
        attribute = self._attributes[item]
        if isinstance(attribute, CompositeAttribute):
            return str(attribute)
        return attribute

    def __setitem__(self, item: str, value: Attribute):
        self._attributes[item] = value

    def __delitem__(self, item: str):
        del self._attributes[item]

    def __iter__(self):
        yield from self._attributes

    def __len__(self):
        return len(self._attributes)

    acc_mode = attribute_accessor("acc-mode")
    deterministic = attribute_accessor("deterministic")
    dropout = attribute_accessor("dropout")
    fpmath = attribute_accessor("fpmath")
    oscale = attribute_accessor("oscale")
    post_ops = attribute_accessor("post-ops")
    rounding_mode = attribute_accessor("rounding-mode")
    scales = attribute_accessor("scales")
    scratchpad = attribute_accessor("scratchpad")
    zero_points = attribute_accessor("zero-points")

    def __str__(self):
        parts = []
        for key, attr in self._attributes.items():
            if isinstance(attr, list):
                sub_parts = "+".join(map(str, attr))
                parts.append(f"{key}:{sub_parts}")
            elif isinstance(attr, dict):
                sub_parts = "+".join(f"{k}:{v!s}" for k, v in attr.items())
                parts.append(f"{key}:{sub_parts}")
            else:
                parts.append(f"{key}:{attr!s}")
        return " ".join(parts)


@dataclass(eq=False)
class HashableEntry(FormattedMapping):
    operation: str
    engine: str
    prim_kind: str
    impl: str
    prop_kind: str
    aux: Dict[str, str]
    mds: List[MemoryDescriptor]
    shapes: str
    exts: Attributes

    def _format(self, convert):
        def stringify(ext):
            if isinstance(ext, list):
                return "+".join(map(convert, ext))
            if isinstance(ext, dict):
                return "+".join(kv_format(k, v) for k, v in ext.items())
            return convert(ext)

        def kv_format(key, value):
            converted = stringify(value)
            if not converted:
                return key
            return f"{key}:{converted}"

        parts = [
            self.operation,
            self.engine,
            self.prim_kind,
            self.impl,
            self.prop_kind,
            " ".join(map(convert, self.mds)),
            " ".join(kv_format(k, v) for k, v in self.exts.items()),
            " ".join(kv_format(k, v) for k, v in self.aux.items()),
            self.shapes,
        ]
        return ",".join(parts)

    def __str__(self):
        return f"onednn_verbose,v1,primitive,{super().__str__()},0"


class Entry(HashableEntry):
    def __init__(
        self,
        *,
        time=0.0,
        timestamp: Optional[float] = None,
        version: int = 0,
        **kwargs,
    ):
        self.time = time
        self.timestamp = timestamp
        self.version = version
        super().__init__(**kwargs)
