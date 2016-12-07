#!/usr/bin/env python
'''
Copyright (c) 2015, Jesper Derehag <jderehag@hotmail.com>
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions
and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import unittest
import argparse
import logging
import subprocess


def _run_tests(args, path, pattern):
    loader = unittest.TestLoader().discover(start_dir=path, pattern=pattern)
    print "Running unittests", pattern, "in", path
    result = unittest.runner.TextTestRunner(descriptions=not args.quicktest).run(loader)
    return result.wasSuccessful()


def _main():
    parser = argparse.ArgumentParser(description="Runner for all unittests", add_help=True)
    parser.add_argument('-v', action="store_true", dest='verbose', default=False, help="enable logger, in DEBUG")
    helptext = "Executes minimum of stuff, suitable for commit-hooks and whatnot"
    parser.add_argument('-q', action="store_true", dest='quicktest', default=False, help=helptext)
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    rootdir = subprocess.check_output('git rev-parse --show-toplevel'.split()).strip()
    if not _run_tests(args, path=rootdir, pattern="*_SUITE.py"):
        print "Unittests failed, fix your crap and rerun", __file__
        exit(-1)

if __name__ == '__main__':
    _main()
