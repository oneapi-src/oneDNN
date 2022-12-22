/*******************************************************************************
* Copyright 2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <cstdio>
#include <map>
#include <string>
#include "utils/json.hpp"
#include "gtest/gtest.h"

TEST(Json, WriterReader) {
    using namespace dnnl::impl::graph;

    std::string filename = "test.txt";
    std::ofstream of(filename);
    utils::json::json_writer_t writer(&of);
    const std::string test_v = "\"\\tr\nlshjk\t\\kv\rm\"";
    ASSERT_NO_THROW(writer.write_string(test_v));
    of.close();
    std::ifstream fs("test.txt");
    utils::json::json_reader_t read(&fs);
    std::string tmp;
    ASSERT_NO_THROW(read.read_string(&tmp));
    ASSERT_NO_THROW(read.read<std::string>(&tmp));

    std::map<std::basic_string<char>, char> mymap;
    utils::json::map_json_t<std::map<std::basic_string<char>, char>> map_tmp;
    ASSERT_NO_THROW(map_tmp.read(&read, &mymap));
    fs.close();
    ASSERT_EQ(std::remove("test.txt"), 0);
}
