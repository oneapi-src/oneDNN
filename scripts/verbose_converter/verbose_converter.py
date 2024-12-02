#!/usr/bin/env python3
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

import argparse
import logging
import sys
from argparse import RawTextHelpFormatter
from typing import IO, Dict, Iterable, List

from src.benchdnn_generator import InputGenerator  # type: ignore
from src.breakdown_generator import BreakdownGenerator  # type: ignore
from src.dnnl_parser import LogParser  # type: ignore
from src.utils import check_version  # type: ignore

default_events = "exec", "create"
stream_handler = logging.StreamHandler(sys.stderr)
fmt = logging.Formatter(fmt="{levelname}: {name}: {message}", style="{")
# workaround for nvim-treesitter indent bug: }
stream_handler.setFormatter(fmt)
logger = logging.getLogger("verbose_converter")
logger.setLevel(logging.CRITICAL + 10)  # off
logger.addHandler(stream_handler)


def one_line(multiline: str):
    return " ".join(map(str.strip, multiline.split("\n"))).strip()


class ConverterError(RuntimeError):
    pass


def generate(generator, parser: LogParser, *args):
    return generator.generate(parser.get_data(), *args)


def convert(
    parser: str,
    input: Iterable[str],
    action: str,
    generator: str,
    split_output: bool,
    agg_keys: List[str],
    events: Iterable[str] = default_events,
) -> Dict[str, str]:
    if not check_version():
        raise ConverterError("Unsupported Python version")

    log_parser: LogParser
    if parser == "oneDNN":
        log_parser = LogParser(logger, input)
    else:
        raise ConverterError("Unsupported parser")

    logger.info("Processing input ...")
    log_parser.process(events)

    if action == "dumpIR":
        logger.info("Dumping data from input...")
        log_parser.dump(True)
        return {}
    elif action == "generate":
        logger.info("Generating output ...")
        if generator == "benchdnn":
            if "create_nested" in events:
                logger.warning(
                    one_line(
                        """
                        Benchdnn arguments generated from create_nested events
                        may not work!
                        """
                    )
                )
            return generate(InputGenerator(logger), log_parser, split_output)
        elif generator == "breakdown":
            return generate(BreakdownGenerator(logger), log_parser, agg_keys)
        else:
            raise ConverterError("Unsupported generator")
    else:
        raise ConverterError("Unsupported action")


def validate_option(value, supported_values, message):
    if value not in supported_values:
        raise ConverterError(message)


def main() -> int:
    if not check_version():
        logger.error("Unsupported Python version")
        return 1

    action_opts = ["generate", "dumpIR"]
    generator_opts = ["benchdnn", "breakdown"]
    parser_opts = ["oneDNN"]
    verbose_opts = [0, 1]
    aggregate_opts = [
        "engine",
        "prim_kind",
        "impl",
        "prop_kind",
        "mds",
        "exts",
        "aux",
        "shapes",
    ]
    event_opts = list(default_events) + ["create_nested"]
    args_parser = argparse.ArgumentParser(
        description="oneDNN log converter", formatter_class=RawTextHelpFormatter
    )
    args_parser.add_argument(
        "-i", "--input", default="stdin", help="input file (default: stdin)"
    )
    args_parser.add_argument(
        "-p",
        "--parser",
        default="oneDNN",
        help=f"type of parser (default: oneDNN). Values: {parser_opts}.",
    )
    args_parser.add_argument(
        "-a",
        "--action",
        default="generate",
        help=f"an action (default: generate). Values: {action_opts}.",
    )
    args_parser.add_argument(
        "-s",
        "--split",
        type=bool,
        default=False,
        help="split generated inputs by primitive kinds (default: False)",
    )
    args_parser.add_argument(
        "-k",
        "--aggregate",
        nargs="+",
        default=aggregate_opts,
        help=one_line(
            f"""
             aggregates statistics on the specified keys (default: all keys but
             time). Values: {aggregate_opts}
             """
        ),
    )
    args_parser.add_argument(
        "-v",
        "--verbose_level",
        default=0,
        type=int,
        help=f"verbose level (default: 0). Values: {verbose_opts}.",
    )
    args_parser.add_argument(
        "-o", "--output", default="stdout", help="output file (default: stdout)"
    )
    args_parser.add_argument(
        "-g",
        "--generator",
        default="benchdnn",
        help=f"target generator (default: benchdnn). Values: {generator_opts}.",
    )
    args_parser.add_argument(
        "-e",
        "--events",
        nargs="+",
        default=list(default_events),
        help=one_line(
            f"""
             events to parse (default: create and exec). Values: {event_opts}.
             """
        ),
    )
    args = args_parser.parse_args()

    # validate options
    logger.setLevel(logging.ERROR)
    try:
        validate_option(args.action, action_opts, "Unknown action value")
        validate_option(
            args.verbose_level, verbose_opts, "Unknown verbose level"
        )
        validate_option(args.parser, parser_opts, "Unknown parser value")
        validate_option(
            args.generator, generator_opts, "Unknown generator value"
        )
        for event in args.events:
            validate_option(event, event_opts, "Unknown event")
    except ConverterError as e:
        logger.error(str(e))
        return 1

    input_data = []
    if args.input == "stdin":
        # if no input was piped, skip reading
        if not sys.stdin.isatty():
            for line in sys.stdin:
                input_data.append(line)
        else:
            logger.warning("No input was provided to the script")
            args_parser.print_help()
    else:
        try:
            input_data = open(args.input, "r").readlines()
        except BaseException as e:
            logger.error(f"While reading input: {e!s}")
            return 1

    event_sets = (
        [[e] for e in args.events]
        if args.generator == "breakdown"
        else [args.events]
    )
    verbosity_levels = [logging.WARNING, logging.INFO]
    logger.setLevel(verbosity_levels[args.verbose_level])

    for events in event_sets:
        try:
            output = convert(
                parser=args.parser,
                input=input_data,
                action=args.action,
                generator=args.generator,
                split_output=args.split,
                agg_keys=args.aggregate,
                events=events,
            )
        except ConverterError as e:
            logger.error(str(e))
            return 1

        for key, value in output.items():
            fd: IO
            filename = args.output
            if args.split:
                filename += f".{key}"
            if args.output != "stdout":
                fd = open(filename, "w")
            else:
                fd = sys.stdout
            if args.generator == "breakdown":
                fd.write(f"Event: {events[0]}\n")
                fd.write(f"{value}\n")
            else:
                if args.split:
                    fd.write(f"--{key}\n")
                fd.write(f"{value}\n")
            if args.output != "stdout":
                fd.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
