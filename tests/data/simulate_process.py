#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from time import sleep

logs_sample = Path(__file__).parent / "taskcluster.log"

with logs_sample.open("r") as f:
    lines = f.readlines()

for line in lines:
    print(line, end="", flush=True)
    sleep(0.01)
