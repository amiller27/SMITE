#!/usr/bin/env python3

from enum import Enum, auto
from pathlib import Path


def fix(line):
    return line.replace(", )", ")").replace(", ]", "]").replace(", }", "}")


def splitlines(line):
    delims = "]})"

    out_line = ""
    for c in line:
        out_line += c
        if c in delims:
            out_line += "\n"

    return out_line


def parse_metis(log):
    class State(Enum):
        PRE_METIS = auto()
        METIS = auto()

    metis_log = [""]
    extras = [""]

    state = State.PRE_METIS

    for line in log.splitlines():
        target = extras

        if state == State.PRE_METIS:
            if line.startswith("     Running `"):
                state = State.METIS
        elif state == State.METIS:
            target = metis_log
        else:
            assert False

        target[0] += splitlines(fix(line)) + "\n"

    return min(metis_log), min(extras)


def parse_smite(log):
    class State(Enum):
        PRE_SMITE = auto()
        SMITE = auto()

    smite_log = [""]
    extras = [""]

    state = State.PRE_SMITE

    for line in log.splitlines():
        target = extras

        if state == State.PRE_SMITE:
            if line.startswith("METIS RESULT: ["):
                state = State.SMITE
        elif state == State.SMITE:
            target = smite_log
        else:
            assert False

        target[0] += splitlines(line) + "\n"

    return min(smite_log), min(extras)


def main(metis_log_path, smite_log_path, extras_log_path):
    metis_log_path = Path(metis_log_path)
    smite_log_path = Path(smite_log_path)
    extras_log_path = Path(extras_log_path)

    metis_log = metis_log_path.read_text()
    smite_log = smite_log_path.read_text()

    # log_size = min(len(metis_log), len(smite_log)) + 1000000
    log_size = int(1e10)

    metis_log, metis_extras = parse_metis(metis_log[:log_size])
    smite_log, smite_extras = parse_smite(smite_log[:log_size])

    metis_log_path.write_text(metis_log)
    smite_log_path.write_text(smite_log)
    extras_log_path.write_text(metis_extras + smite_extras)


if __name__ == "__main__":
    import argh

    argh.dispatch_command(main)
