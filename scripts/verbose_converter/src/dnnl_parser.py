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
from typing import Iterable, List, Tuple

from . import ir, parse


class LoggingContext:
    def __init__(self, logger):
        self.logger = logger

    def __enter__(self):
        return self

    def __exit__(self, type, value, _):
        if type is not None and issubclass(type, parse.ParseError):
            self.logger.warning(str(value))
            return True


class LogParser:
    """
    Parses a log file with oneDNN verbose and converts it into internal
    representation.
    """

    def __init__(self, logger, input: Iterable[str] = ()):
        self.input = input
        self.error_handler = LoggingContext(logger)
        self.data: List[Tuple[str, ir.Entry]] = []

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

        parser = parse.Parser(self.input, filter_events, self.error_handler)
        self.data = list(parser)

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

        return {i: entry for i, (_, entry) in enumerate(self.data)}

    def dump(self, converted=False):
        """
        Prints data parsed from input to stdout.

        Parameters
        ----------
        converted (default: False) -- If truthy, prints data in internal
        representation, otherwise prints data in the original form.

        Returns
        -------
        None
        """

        for i, (line, entry) in enumerate(self.data):
            if converted:
                print(f"{i}, {entry!r}")
            else:
                print(line)
