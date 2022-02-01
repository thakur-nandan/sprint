# Packaging pyserini.index into both callable and entry point
# With reference to https://github.com/castorini/pyserini/blob/master/pyserini/search/__main__.py
#
# Pyserini: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import argparse
import inspect
import pyserini.index._base  # important to import this for starting Java services
from jnius import autoclass  # this line should go after


def run(
    collection: str,
    input: str,
    index: str,
    generator: str,
    impact: bool,
    pretokenized: bool,
    threads: int
):
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    assert os.path.isdir(input), ValueError('Argument -input should be a directory.')
    args_strings = []
    for arg in args:
        value = values[arg]
        args_strings.append(f'-{arg}')
        if type(value) is not bool:
            args_strings.append(str(value))
    
    JIndexCollection = autoclass('io.anserini.index.IndexCollection')
    JIndexCollection.main(args_strings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', '-collection')
    parser.add_argument('--input', '-input')
    parser.add_argument('--index', '-index')
    parser.add_argument('--generator', '-generator')
    parser.add_argument('--impact', '-impact', action='store_true')
    parser.add_argument('--pretokenized', '-pretokenized', action='store_true')
    parser.add_argument('--threads', '-threads')
    args = parser.parse_args()
    run(**vars(args))
