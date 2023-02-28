/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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
#ifndef GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_EXCEPTION_UTIL_HPP
#define GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_EXCEPTION_UTIL_HPP

#include <stdexcept>
#include <string>

#define EXPECT_EXCEPTION(TRY_BLOCK, EXCEPTION_TYPE, MESSAGE) \
    try { \
        TRY_BLOCK; \
        FAIL() << "Expecting an exception!"; \
    } catch (const EXCEPTION_TYPE &e) { \
        EXPECT_TRUE(std::string(e.what()).find(MESSAGE) != std::string::npos) \
                << " Wrong exception message: " << e.what() \
                << "\nExpecting:" << MESSAGE; \
    } catch (...) { FAIL() << " Wrong exception type!"; }

#define EXPECT_SC_ERROR(CODE, MESSAGE) \
    EXPECT_EXCEPTION(CODE, ::std::exception, MESSAGE)

#endif
