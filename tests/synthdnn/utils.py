################################################################################
# Copyright 2024 Intel Corporation
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
import sys
import os

def DebugPrint(msg=""):
    frame = sys._getframe().f_back
    fname = frame.f_code.co_filename
    head, tail = os.path.split(fname)
    print(f"Debug: {tail}:{sys._getframe().f_back.f_lineno}: {msg}")

def getenv(var):
    return os.environ.get(var)

