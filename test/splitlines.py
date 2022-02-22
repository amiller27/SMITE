#!/usr/bin/env python3
import sys

delims = "]})"

for c in sys.stdin.read():
    sys.stdout.write(c)
    if c in delims:
        sys.stdout.write("\n")
