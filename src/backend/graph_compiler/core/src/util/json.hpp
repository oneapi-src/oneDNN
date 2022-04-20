/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_JSON_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_JSON_HPP

#include <algorithm>
#include <cassert>
#include <cctype>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <util/general_object.hpp>

#if defined(__GNUC__)
#define ATTRIBUTE_UNUSED __attribute__((unused))
#else
#define ATTRIBUTE_UNUSED
#endif
#define STR_CONCAT_(__x, __y) __x##__y
#define STR_CONCAT(__x, __y) STR_CONCAT_(__x, __y)
namespace sc {

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

struct anyhandler;
/*!
 * \brief generic serialization handler
 * \tparam T the type to be serialized
 */
template <typename T>
struct handler;

template <typename T>
struct common_handler;

template <typename T>
struct reflection_handler;

/*!
 * \brief whether a type is arithemetic type
 * \tparam T the type to query
 */
// template <typename T>
// struct is_arithmetic {
//     static const bool value = std::is_arithmetic<T>::value;
// };

/*!
 * \brief Lightweight json to write any STL compositions.
 */
class SC_INTERNAL_API json_writer {
public:
    /*!
     * \brief Constructor.
     * \param os the output reciever.
     * \param pretty_print if printing the pretty format
     * \param skip_temp skip writing key-value pairs with key starting with
     * "temp."
     */
    json_writer(
            std::ostream *os, bool pretty_print = true, bool skip_temp = false)
        : os_(os), pretty_print_(pretty_print), skip_temp_(skip_temp) {}
    /*!
     * \brief Start beginning of array.
     * \param multi_line whether to start an multi_line array.
     */
    inline void begin_object(bool multi_line = true);
    /*! \brief Finish writing object. */
    inline void end_object();
    /*!
     * \brief write key value pair in the object.
     * \param key the key of the object.
     * \param value the value of to be written.
     * \tparam ValueType The value type to be written.
     */
    template <typename ValueType>
    inline void write_keyvalue(const std::string &key, const ValueType &value);

    /*
     * \brief write a string that can contain escape characters.
     * \param v the value to be written.
     * \tparam ValueType The value type to be written.
     */
    template <typename ValueType>
    inline void write_number(const ValueType &v);

    /*
     * \brief write a bool
     * \param v the value to be wirtten
     */
    inline void write_bool(const bool &v);
    /*!
     * \brief write a string that can contain escape characters.
     * \param s the string to be written.
     */
    inline void write_string(const std::string &s);
    template <typename ValueType>
    inline void write(const ValueType &value);
    /*!
     * \brief Start beginning of array.
     * \param multi_line whether to start an multi_line array.
     */
    inline void begin_array(bool multi_line = true);
    /*! \brief Finish writing an array. */
    inline void end_array();
    /*!
     * \brief write seperator of array, before writing next element.
     * User can proceed to call writer->write to write next item
     */
    inline void write_array_seperator();
    /*!
     * \brief write seperator of item, before writing next item
     */
    inline void write_item_seperator();
    /*!
     * \brief write value into array.
     * \param value The value of to be written.
     * \tparam ValueType The value type to be written.
     */
    template <typename ValueType>
    inline void write_array_item(const ValueType &value);

    void write_key(const std::string &key) {
        if (scope_counter_.back() > 0) { *os_ << ","; }
        write_seperator();
        *os_ << '\"';
        *os_ << key;
        *os_ << "\": ";
        scope_counter_.back() += 1;
    }

private:
    std::ostream *os_;
    // if using pretty printing
    bool pretty_print_;
    // if skipping "temp.*" keys
    bool skip_temp_;
    /*!
     * \brief record how many element processed in
     *  current array/object scope.
     */
    std::vector<size_t> scope_counter_;
    /*! \brief Record whether current is a multiline scope */
    std::vector<bool> scope_multi_line_;
    /*!
     * \brief write seperating space and newlines
     */
    inline void write_seperator();
};

class SC_INTERNAL_API json_reader {
public:
    /*!
     * \brief Constructor.
     * \param is the input source.
     */
    explicit json_reader(std::istream *is)
        : is_(is), line_count_r_(0), line_count_n_(0) {}
    /*!
     * \brief Parse next JSON string.
     * \param out_str the output string.
     */
    inline void read_string(std::string *out_str);
    /*!
     * \brief read Number.
     * \param out_value output value;
     * \tparam ValueType type of the number
     */
    template <typename ValueType>
    inline void read_number(ValueType *out_value);
    /*
     * \brief read a bool value
     * \param v the value to be read
     */
    inline void read_bool(bool *out_value);
    /*!
     * \brief Begin parsing an object.
     */
    inline void begin_object();
    /*!
     * \brief Begin parsing an array.
     */
    inline void begin_array();
    /*!
     * \brief Try to move to next object item.
     *  If this call is successful, user can proceed to call
     *  reader->read to read in the value.
     * \param out_key the key to the next object.
     * \return true if the read is successful, false if we are at end of the
     * object.
     */
    inline bool next_object_item(std::string *out_key);
    /*!
     * \brief Try to read the next element in the array.
     *  If this call is successful, user can proceed to call
     *  reader->read to read in the value.
     * \return true if the read is successful, false if we are at end of the
     * array.
     */
    inline bool next_array_item();
    /*!
     * \brief read next ValueType.
     * \param out_value any STL or json readable type to be read
     * \tparam ValueType the data type to be read.
     */
    template <typename ValueType>
    inline void read(ValueType *out_value);
    /*!
     * \brief read just before next nonspace but not read that.
     * \return the next nonspace character.
     */
    inline int peeknext_nonspace();
    /*!
     *\brief set field type for any structure
     */
    inline void add_field_type(
            const std::unordered_map<std::string, std::string> &field_type) {
        field_type_.insert(field_type.begin(), field_type.end());
    }
    inline std::unordered_map<std::string, std::string> get_field_type() {
        return field_type_;
    }

    void expect_next_object_key(const std::string &expected_key);

    void expect_object_ends();

private:
    /*! \brief internal reader stream */
    std::istream *is_;
    /*! \brief "\\r" counter */
    size_t line_count_r_;
    /*! \brief "\\n" counter */
    size_t line_count_n_;
    /*!
     * \brief record how many element processed in
     *  current array/object scope.
     */
    std::vector<size_t> scope_counter_;
    /*!
     * \brief record field type for any structrue
     */
    std::unordered_map<std::string, std::string> field_type_;
    /*!
     * \brief read next nonspace character.
     * \return the next nonspace character.
     */
    inline int next_nonspace();
    // /*!
    //  * \brief read just before next nonspace but not read that.
    //  * \return the next nonspace character.
    //  */
    // inline int peeknext_nonspace();
    /*!
     * \brief Takes the next char from the input source.
     * \return the next character.
     */
    inline int next_char();
    /*!
     * \brief Returns the next char from the input source.
     * \return the next character.
     */
    inline int peeknext_char();
};

class readhelper {
public:
    /*!
     * \brief Declare field of type T
     * \param key the key of the of field.
     * \param addr address of the data type.
     * \tparam T the data type to be read, must be STL composition of JSON
     * serializable.
     */
    template <typename T>
    inline void declare_field(const std::string &key, T *addr) {
        if (map_.count(key) == 0) {
            entry e;
            e.func = reader_function<T>;
            e.addr = static_cast<void *>(addr);
            map_[key] = e;
        }
    }
    /*!
     * \brief read in all the declared fields.
     * \param reader the reader to read the json.
     */
    inline void read_fields(json_reader *reader);

private:
    /*!
     * \brief The internal reader function.
     * \param reader The reader to read.
     * \param addr The memory address to read.
     */
    template <typename T>
    inline static void reader_function(json_reader *reader, void *addr);
    /*! \brief callback type to reader function */
    typedef void (*readfunction)(json_reader *reader, void *addr);
    /*! \brief internal data entry */
    struct entry {
        /*! \brief the reader function */
        readfunction func;
        /*! \brief the address to read */
        void *addr;
    };
    /*! \brief the internal map of reader callbacks */
    std::map<std::string, entry> map_;
    std::unordered_map<std::string, std::string> field_type_;
};

template <typename Then, typename Else>
struct if_else_type<true, Then, Else> {
    typedef Then Type;
};

template <typename Then, typename Else>
struct if_else_type<false, Then, Else> {
    typedef Else Type;
};

template <typename ValueType>
struct enum_handler {
    inline static void write(json_writer *writer, const ValueType &value) {
        writer->write_number<int32_t>(static_cast<int32_t>(value));
    }
    inline static void read(json_reader *reader, ValueType *value) {
        int32_t tmp_value;
        reader->read_number<int32_t>(&tmp_value);
        *value = static_cast<ValueType>(tmp_value);
    }
};

template <typename ValueType>
struct numeric_handler {
    inline static void write(json_writer *writer, const ValueType &value) {
        writer->write_number<ValueType>(value);
    }
    inline static void read(json_reader *reader, ValueType *value) {
        reader->read_number<ValueType>(value);
    }
};

template <typename T>
struct common_handler {
    using TReal = typename std::decay<T>::type;
    inline static void write(json_writer *writer, const T &value) {
        typedef typename if_else_type<std::is_enum<TReal>::value,
                enum_handler<TReal>, handler<TReal>>::Type t_handler;
        t_handler::write(writer, value);
    }
    inline static void read(json_reader *reader, TReal *value) {
        typedef typename if_else_type<std::is_enum<TReal>::value,
                enum_handler<TReal>, handler<TReal>>::Type t_handler;
        t_handler::read(reader, value);
    }
};

template <typename ValueType>
struct common_handler<std::shared_ptr<ValueType>> {
    inline static void write(
            json_writer *writer, const std::shared_ptr<ValueType> &value) {
        writer->write(*value);
    }

    inline static void read(
            json_reader *reader, std::shared_ptr<ValueType> *value) {
        *value = std::make_shared<ValueType>();
        reader->read(value->get());
    }
};

template <typename K, typename V>
struct handler<std::pair<K, V>> {
    inline static void write(json_writer *writer, const std::pair<K, V> &kv) {
        writer->begin_array(false);
        writer->write_array_item(kv.first);
        writer->write_array_item(kv.second);
        writer->end_array();
    }
    inline static void read(json_reader *reader, std::pair<K, V> *kv) {
        reader->begin_array();
        if (reader->next_array_item() != 1)
            throw std::runtime_error("JSON: Expect array of length 2");
        handler<K>::read(reader, &(kv->first));
        if (reader->next_array_item() != 1)
            throw std::runtime_error("JSON: Expect array of length 2");
        handler<V>::read(reader, &(kv->second));
        if (reader->next_array_item() != 0)
            throw std::runtime_error("JSON: Expect array of length 2");
    }
};

template <typename ContainerType,
        typename ElemType = typename ContainerType::value_type>
struct arrayhandler {
    inline static void write(json_writer *writer, const ContainerType &array) {
        // typedef typename ContainerType::value_type ElemType;
        writer->begin_array(array.size() > 5);
        for (typename ContainerType::const_iterator it = array.begin();
                it != array.end(); ++it) {
            writer->write_array_item(*it);
        }
        writer->end_array();
    }
    inline static void read(json_reader *reader, ContainerType *array) {
        array->clear();
        reader->begin_array();
        while (reader->next_array_item()) {
            ElemType value;
            handler<ElemType>::read(reader, &value);
            array->insert(array->end(), std::move(value));
        }
    }
};

template <typename ContainerType>
struct maphandler {
    inline static void write(json_writer *writer, const ContainerType &map) {
        writer->begin_object(map.size() > 1);
        for (typename ContainerType::const_iterator it = map.begin();
                it != map.end(); ++it) {
            writer->write_keyvalue(it->first, it->second);
        }
        writer->end_object();
    }
    inline static void read(json_reader *reader, ContainerType *map) {
        typedef typename ContainerType::mapped_type ElemType;
        map->clear();
        reader->begin_object();
        std::string key;
        while (reader->next_object_item(&key)) {
            ElemType value;
            reader->read(&value);
            (*map)[key] = std::move(value);
        }
    }
};

template <>
struct handler<std::string> {
    inline static void write(json_writer *writer, const std::string &value) {
        writer->write_string(value);
    }
    inline static void read(json_reader *reader, std::string *str) {
        reader->read_string(str);
    }
};

template <>
struct handler<bool> {
    inline static void write(json_writer *writer, const bool &value) {
        writer->write_bool(value);
    }
    inline static void read(json_reader *reader, bool *str) {
        reader->read_bool(str);
    }
};

template <typename T>
struct handler<std::vector<T>> : public arrayhandler<std::vector<T>> {};

template <typename V>
struct handler<std::map<std::string, V>>
    : public maphandler<std::map<std::string, V>> {};

template <typename V>
struct handler<std::unordered_map<std::string, V>>
    : public maphandler<std::unordered_map<std::string, V>> {};

template <typename K, typename V>
struct handler<std::unordered_map<K, V>>
    : public arrayhandler<std::unordered_map<K, V>, std::pair<K, V>> {};

template <typename V>
struct handler<std::unordered_set<V>>
    : public arrayhandler<std::unordered_set<V>> {};

/*!
 * \brief generic serialization handler
 * \tparam T the type to be serialized
 */
template <typename T>
struct SC_INTERNAL_API handler {
    typedef typename if_else_type<std::is_arithmetic<T>::value,
            numeric_handler<T>, common_handler<T>>::Type numeric_or_common;
    typedef typename if_else_type<
            std::is_base_of<reflection::reflection_enabled_t, T>::value,
            reflection_handler<T>, numeric_or_common>::Type t_handler;
    inline static void write(json_writer *writer, const T &data) {
        t_handler::write(writer, data);
    }
    inline static void read(json_reader *reader, T *data) {
        t_handler::read(reader, data);
    }
};

template <>
struct SC_INTERNAL_API handler<any_t> {
    static void write(json_writer *writer, const any_t &data);
    static void read(json_reader *reader, any_t *data);
};

template <>
struct SC_INTERNAL_API handler<reflection::general_ref_t> {
    static void write(
            json_writer *writer, const reflection::general_ref_t &data);
    static void read(json_reader *reader, reflection::general_ref_t *data);
};

template <>
struct SC_INTERNAL_API handler<reflection::general_object_t> {
    static void write(
            json_writer *writer, const reflection::general_object_t &data);
    static void read(json_reader *reader, reflection::general_object_t *data);
};

template <typename T>
struct SC_INTERNAL_API reflection_handler {
    static void write(json_writer *writer, const T &data) {
        writer->write(reflection::general_ref_t::from(data));
    }
    static void read(json_reader *reader, T *data) {
        auto ref = reflection::general_ref_t::from(*data);
        reader->read(&ref);
    }
};

inline void json_writer::begin_object(bool multi_line) {
    *os_ << "{";
    scope_multi_line_.push_back(multi_line);
    scope_counter_.push_back(0);
}

template <typename ValueType>
inline void json_writer::write_number(const ValueType &v) {
    *os_ << v;
}

inline void json_writer::write_bool(const bool &v) {
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
    scope_counter_.push_back(0);
}

template <typename ValueType>
inline void json_writer::write_keyvalue(
        const std::string &key, const ValueType &value) {
    if (skip_temp_ && key.find("temp.") == 0) { return; }
    write_key(key);
    json::handler<ValueType>::write(this, value);
}

inline void json_writer::end_array() {
    if (scope_counter_.size() != 0 && scope_multi_line_.size() != 0) {
        bool newline = scope_multi_line_.back();
        size_t nelem = scope_counter_.back();
        scope_multi_line_.pop_back();
        scope_counter_.pop_back();
        if (newline && nelem != 0) write_seperator();
    }
    *os_ << ']';
}

inline void json_writer::write_array_seperator() {
    if (scope_counter_.back() != 0) { *os_ << ", "; }
    scope_counter_.back() += 1;
    write_seperator();
}

inline void json_writer::write_item_seperator() {
    *os_ << ", ";
    write_seperator();
}
template <typename ValueType>
inline void json_writer::write_array_item(const ValueType &value) {
    this->write_array_seperator();
    json::handler<ValueType>::write(this, value);
}

inline void json_writer::end_object() {
    // TODO(Zhichen): Replace comparison by log check
    if (scope_counter_.size() != 0 && scope_multi_line_.size() != 0) {
        bool newline = scope_multi_line_.back();
        size_t nelem = scope_counter_.back();
        scope_multi_line_.pop_back();
        scope_counter_.pop_back();
        if (newline && nelem != 0) write_seperator();
        *os_ << '}';
    }
}

template <typename ValueType>
inline void json_writer::write(const ValueType &value) {
    size_t nscope = scope_multi_line_.size();
    handler<ValueType>::write(this, value);
    assert(nscope == scope_multi_line_.size());
}

inline void json_writer::write_seperator() {
    if (pretty_print_
            && (scope_multi_line_.size() == 0 || scope_multi_line_.back())) {
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
        if (ch == '\n') ++line_count_n_;
        if (ch == '\r') ++line_count_r_;
    } while (isspace(ch));
    return ch;
}

inline int json_reader::peeknext_nonspace() {
    int ch;
    while (true) {
        ch = peeknext_char();
        if (ch == '\n') ++line_count_n_;
        if (ch == '\r') ++line_count_r_;
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
                    case 'r': output << '\r'; break;
                    case 'n': output << '\n'; break;
                    case '\\': output << '\\'; break;
                    case 't': output << '\t'; break;
                    case '\"': output << '\"'; break;
                    default:
                        throw std::runtime_error(
                                "JSON: unknown string escape.");
                }
            } else {
                if (ch == '\"') break;
                output << static_cast<char>(ch);
            }
            if (ch == EOF || ch == '\r' || ch == '\n') {
                throw std::runtime_error("JSON: error at!");
                return;
            }
        }
        *out_str = output.str();
    } else {
        return;
    }
}

template <typename ValueType>
inline void json_reader::read_number(ValueType *out_value) {
    *is_ >> *out_value;
}

inline void json_reader::read_bool(bool *out_value) {
    *is_ >> *out_value;
}

inline void json_reader::begin_object() {
    int ch = next_nonspace();
    if (ch == '{') { scope_counter_.push_back(0); }
}

inline void json_reader::begin_array() {
    int ch = next_nonspace();
    if (ch == '[') { scope_counter_.push_back(0); }
}

inline bool json_reader::next_object_item(std::string *out_key) {
    bool next = true;
    if (scope_counter_.back() != 0) {
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
        scope_counter_.pop_back();
        return false;
    } else {
        scope_counter_.back() += 1;
        read_string(out_key);
        int ch = next_nonspace();
        return (ch == ':');
    }
}

inline bool json_reader::next_array_item() {
    bool next = true;
    if (scope_counter_.back() != 0) {
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
        scope_counter_.pop_back();
        return false;
    } else {
        scope_counter_.back() += 1;
        return true;
    }
}

template <typename T>
inline void json_reader::read(T *out_value) {
    json::handler<T>::read(this, out_value);
}

inline void readhelper::read_fields(json_reader *reader) {
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
                    && "json reader: missing filed ");
        }
    }
}

template <typename T>
inline void readhelper::reader_function(json_reader *reader, void *addr) {
    json::handler<T>::read(reader, static_cast<T *>(addr));
}

} // namespace json
} // namespace sc

#endif
