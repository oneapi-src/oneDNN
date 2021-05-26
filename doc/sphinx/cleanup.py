################################################################################
# Copyright 2021 Intel Corporation
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

import sys, os

DIR = sys.argv[1]

for root, dirs, files in os.walk(DIR):
    for file in files:
        if file.endswith(".rst"):
            # XXX: A hack for WSL. Based on setup WSL might not take into
            # account case sensitivity, and as a result doxygen might generate
            # uppercased files.
            # Temp file should be used in order to make files lowcased, because
            # direct renaming doesn't work due to case insensitive file system.
            if file.lower() != file:
                tmp_file = "tmp_" + file
                os.rename(os.path.join(root, file), \
                        os.path.join(root, tmp_file))
                os.rename(os.path.join(root, tmp_file), \
                        os.path.join(root, file.lower()))
            if file.startswith("page_dev_guide"):
                stripped_file = file[5:]
                # if destination file already exists then remove the source file
                if stripped_file not in files:
                    os.rename(os.path.join(root, file), \
                            os.path.join(root, stripped_file))
                else:
                    os.remove(os.path.join(root, file))
