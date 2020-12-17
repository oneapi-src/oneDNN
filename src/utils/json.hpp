/*******************************************************************************
 * Copyright 2020 Intel Corporation
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

#ifndef LLGA_UTILS_JSON_HPP
#define LLGA_UTILS_JSON_HPP

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace llga {
namespace impl {
namespace utils {
namespace json {

/*!
 * \brief template to select type based on condition
 * For example, if_else_type<true, int, float>::Type will give int
 * \tparam cond the condition
 * \tparam Then the typename to be returned if cond is true
 * \tparam Else typename to be returned if cond is false
 */

template <bool cond, typename Then, typename Else>
struct if_else_type;

/*!
 * \brief generic serialization json
 * \tparam T the type to be serialized
 */
template <typename T>
struct json_handler;

template <typename T>
struct common_json;

/*!
 * \brief json to write any type.
 */
class json_writer {
public:
    json_writer(std::ostream *os) : os_(os) {}
    /*!
     * \brief object begin
     * \param multi_line whether to start an multi_line array.
     */
    inline void begin_object();
    /*! \brief object end. */
    inline void end_object();
    /*!
     * \brief write key value pair in the object.
     */
    template <typename valuetype>
    inline void write_keyvalue(const std::string &key, const valuetype &value);
    /*!
     * \brief write a number.
     */
    template <typename valuetype>
    inline void write_number(const valuetype &v);
    /*!
     * \brief write a string.
     */
    inline void write_string(const std::string &s);
    /*!
     * \brief array begin.
     * \param multi_line if true, write multi_line.
     */
    inline void begin_array(bool multi_line = true);
    /*! \brief array end. */
    inline void end_array();
    /*!
     * \brief write array sepaerator.
     */
    inline void write_array_seperator();
    /*!
     * \brief write array value.
     */
    template <typename valuetype>
    inline void write_array_item(const valuetype &value);

private:
    std::ostream *os_;
    /*!
     * \brief record how many element in the currentscope.
     */
    std::vector<size_t> scope_count_;
    /*! \brief record if current pos is a multiline scope */
    std::vector<bool> scope_multi_line_;
    /*!
     * \brief write seperating space and newlines
     */
    inline void write_seperator();
};

class json_reader {
public:
    explicit json_reader(std::istream *is) : is_(is) {}
    /*!
     * \brief parse json string.
     */
    inline void read_string(std::string *out_str);
    /*!
     * \brief read number.
     */
    template <typename valuetype>
    inline void read_number(valuetype *out_value);
    /*!
     * \brief parse an object begin.
     */
    inline void begin_object();
    /*!
     * \brief parse an object end.
     */
    inline void begin_array();
    /*!
     * \brief read next object, if true, will read next object.
     */
    inline bool next_object_item(std::string *out_key);
    /*!
     * \brief read next object, if true, will read next object.
     */
    inline bool next_array_item();
    /*!
     * \brief read next value.
     */
    template <typename valuetype>
    inline void read(valuetype *out_value);

private:
    std::istream *is_;
    /*!
     * \brief record element size in the current.
     */
    std::vector<size_t> scope_count_;
    /*!
     * \brief read next nonspace char.
     */
    inline int next_nonspace();
    inline int peeknext_nonspace();
    /*!
   * \brief get the next char from the input.
   */
    inline int next_char();
    inline int peeknext_char();
};

class read_helper {
public:
    /*!
   * \brief declare field
   */
    template <typename T>
    inline void declare_field(const std::string &key, T *addr) {
        //declare_fieldInternal(key, addr);
        if (map_.count(key) == 0) {
            entry e;
            e.func = reader_function<T>;
            e.addr = static_cast<void *>(addr);
            map_[key] = e;
        }
    }
    /*!
   * \brief read all fields according to declare.
   */
    inline void read_fields(json_reader *reader);

private:
    /*!
     * \brief reader function to store T.
     */
    template <typename T>
    inline static void reader_function(json_reader *reader, void *addr);
    /*! \brief callback type to reader function */
    typedef void (*readfunc)(json_reader *reader, void *addr);
    /*! \brief data entry */
    struct entry {
        readfunc func;
        /*! \brief store the address data for reading json*/
        void *addr;
    };
    /*! \brief reader callback */
    std::map<std::string, entry> map_;
};

template <typename then, typename other>
struct if_else_type<true, then, other> {
    typedef then type;
};

template <typename then, typename other>
struct if_else_type<false, then, other> {
    typedef other type;
};

template <typename valuetype>
struct num_json {
    inline static void write(json_writer *writer, const valuetype &value) {
        writer->write_number<valuetype>(value);
    }
    inline static void read(json_reader *reader, valuetype *value) {
        reader->read_number<valuetype>(value);
    }
};

template <typename valuetype>
struct common_json {
    inline static void write(json_writer *writer, const valuetype &value) {
        value.save(writer);
    }
    inline static void read(json_reader *reader, valuetype *value) {
        value->load(reader);
    }
};

template <typename valuetype>
struct common_json<std::shared_ptr<valuetype>> {
    inline static void write(
            json_writer *writer, const std::shared_ptr<valuetype> &value) {
        auto *v = value.get();
        v->save(writer);
    }

    inline static void read(
            json_reader *reader, std::shared_ptr<valuetype> *value) {
        auto ptr = std::make_shared<valuetype>();
        auto *v = ptr.get();
        v->load(reader);
        *value = ptr;
    }
};

template <typename CT>
struct array_json {
    inline static void write(json_writer *writer, const CT &array) {
        writer->begin_array();
        for (typename CT::const_iterator it = array.begin(); it != array.end();
                ++it) {
            writer->write_array_item(*it);
        }
        writer->end_array();
    }
    inline static void read(json_reader *reader, CT *array) {
        typedef typename CT::value_type elemtype;
        array->clear();
        reader->begin_array();
        while (reader->next_array_item()) {
            elemtype value;
            json_handler<elemtype>::read(reader, &value);
            array->insert(array->end(), value);
        }
    }
};

template <>
struct json_handler<std::string> {
    inline static void write(json_writer *writer, const std::string &value) {
        writer->write_string(value);
    }
    inline static void read(json_reader *reader, std::string *str) {
        reader->read_string(str);
    }
};

template <typename T>
struct json_handler<std::vector<T>> : public array_json<std::vector<T>> {};

template <typename T>
struct json_handler<std::list<T>> : public array_json<std::list<T>> {};
/*!
 * \brief generic serialization json
 */
template <typename T>
struct json_handler {
    inline static void write(json_writer *writer, const T &data) {
        typedef typename if_else_type<std::is_arithmetic<T>::value, num_json<T>,
                common_json<T>>::type Tjson;
        Tjson::write(writer, data);
    }
    inline static void read(json_reader *reader, T *data) {
        typedef typename if_else_type<std::is_arithmetic<T>::value, num_json<T>,
                common_json<T>>::type Tjson;
        Tjson::read(reader, data);
    }
};

inline void json_writer::begin_object() {
    *os_ << "{";
    scope_count_.push_back(0);
}

template <typename valuetype>
inline void json_writer::write_keyvalue(
        const std::string &key, const valuetype &value) {
    if (scope_count_.back() > 0) { *os_ << ","; }
    write_seperator();
    *os_ << '\"';
    *os_ << key;
    *os_ << "\": ";
    scope_count_.back() += 1;
    json_handler<valuetype>::write(this, value);
}

template <typename valuetype>
inline void json_writer::write_number(const valuetype &v) {
    *os_ << v;
}

inline void json_writer::write_string(const std::string &s) {
    *os_ << '\"';
    for (size_t i = 0; i < s.length(); ++i) {
        char ch = s[i];
        switch (ch) {
            case '\r': *os_ << "\\r"; break;
            case '\n': *os_ << "\\n"; break;
            case '\\': *os_ << "\\\\"; break;
            case '\t': *os_ << "\\t"; break;
            case '\"': *os_ << "\\\""; break;
            default: *os_ << ch;
        }
    }
    *os_ << '\"';
}

inline void json_writer::begin_array(bool multi_line) {
    *os_ << '[';
    scope_multi_line_.push_back(multi_line);
    scope_count_.push_back(0);
}

inline void json_writer::end_array() {
    if (scope_count_.size() != 0 && scope_multi_line_.size() != 0) {
        bool newline = scope_multi_line_.back();
        size_t nelem = scope_count_.back();
        scope_multi_line_.pop_back();
        scope_count_.pop_back();
        if (newline && nelem != 0) write_seperator();
    }
    *os_ << ']';
}

inline void json_writer::write_array_seperator() {
    if (scope_count_.back() != 0) { *os_ << ", "; }
    scope_count_.back() += 1;
    write_seperator();
}

template <typename valuetype>
inline void json_writer::write_array_item(const valuetype &value) {
    this->write_array_seperator();
    json::json_handler<valuetype>::write(this, value);
}

inline void json_writer::end_object() {
    if (scope_count_.size() != 0) {
        size_t nelem = scope_count_.back();
        scope_count_.pop_back();
        if (nelem != 0) write_seperator();
        *os_ << '}';
    }
}

inline void json_writer::write_seperator() {
    if (scope_multi_line_.size() == 0 || scope_multi_line_.back()) {
        *os_ << '\n';
        *os_ << std::string(scope_multi_line_.size() * 2, ' ');
    }
}

inline int json_reader::next_char() {
    return is_->get();
}

inline int json_reader::peeknext_char() {
    return is_->peek();
}

inline int json_reader::next_nonspace() {
    int ch;
    do {
        ch = next_char();
    } while (isspace(ch));
    return ch;
}

inline int json_reader::peeknext_nonspace() {
    int ch;
    while (true) {
        ch = peeknext_char();
        if (!isspace(ch)) break;
        next_char();
    }
    return ch;
}

inline void json_reader::read_string(std::string *out_str) {
    int ch = next_nonspace();
    if (ch == '\"') {
        std::ostringstream output;
        while (true) {
            ch = next_char();
            if (ch == '\\') {
                char sch = static_cast<char>(next_char());
                switch (sch) {
                    case 'r': output << "\r"; break;
                    case 'n': output << "\r"; break;
                    case '\\': output << "\r"; break;
                    case 't': output << "\r"; break;
                    case '\"': output << "\r"; break;
                    default: throw("unknown string escape.");
                }
            } else {
                if (ch == '\"') break;
                output << static_cast<char>(ch);
            }
            if (ch == EOF || ch == '\r' || ch == '\n') {
                throw("error at!");
                return;
            }
        }
        *out_str = output.str();
    }
}

template <typename valuetype>
inline void json_reader::read_number(valuetype *out_value) {
    *is_ >> *out_value;
}

inline void json_reader::begin_object() {
    int ch = next_nonspace();
    if (ch == '{') { scope_count_.push_back(0); }
}

inline void json_reader::begin_array() {
    int ch = next_nonspace();
    if (ch == '[') { scope_count_.push_back(0); }
}

inline bool json_reader::next_object_item(std::string *out_key) {
    bool next = true;
    if (scope_count_.back() != 0) {
        int ch = next_nonspace();
        if (ch == EOF) {
            next = false;
        } else if (ch == '}') {
            next = false;
        } else {
            assert(ch == ',');
        }
    } else {
        int ch = peeknext_nonspace();
        if (ch == '}') {
            next_char();
            next = false;
        }
    }
    if (!next) {
        scope_count_.pop_back();
        return false;
    } else {
        scope_count_.back() += 1;
        read_string(out_key);
        int ch = next_nonspace();
        return (ch == ':');
    }
}

inline bool json_reader::next_array_item() {
    bool next = true;
    if (scope_count_.back() != 0) {
        int ch = next_nonspace();
        if (ch == EOF) {
            next = false;
        } else if (ch == ']') {
            next = false;
        } else {
            assert(ch == ',');
        }
    } else {
        int ch = peeknext_nonspace();
        if (ch == ']') {
            next_char();
            next = false;
        }
    }
    if (!next) {
        scope_count_.pop_back();
        return false;
    } else {
        scope_count_.back() += 1;
        return true;
    }
}

template <typename valuetype>
inline void json_reader::read(valuetype *out_value) {
    json::json_handler<valuetype>::read(this, out_value);
}

inline void read_helper::read_fields(json_reader *reader) {
    reader->begin_object();
    std::map<std::string, int> visited;
    std::string key;
    while (reader->next_object_item(&key)) {
        if (map_.count(key) != 0) {
            entry e = map_[key];
            (*e.func)(reader, e.addr);
            visited[key] = 0;
        }
    }
    if (visited.size() != map_.size()) {
        for (std::map<std::string, entry>::iterator it = map_.begin();
                it != map_.end(); ++it) {
            assert(visited.count(it->first) == 1
                    && "json reader: missing field");
        }
    }
}

template <typename T>
inline void read_helper::reader_function(json_reader *reader, void *addr) {
    json::json_handler<T>::read(reader, static_cast<T *>(addr));
}

} // namespace json
} // namespace utils
} // namespace impl
} // namespace llga

#endif
