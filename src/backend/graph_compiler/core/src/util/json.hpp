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
#include <util/exceptions.hpp>
#include <util/general_object.hpp>

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
    void begin_object(bool multi_line = true);
    /*! \brief Finish writing object. */
    void end_object();
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
    void write_bool(const bool &v);
    /*!
     * \brief write a string that can contain escape characters.
     * \param s the string to be written.
     */
    void write_string(const std::string &s);
    template <typename ValueType>
    inline void write(const ValueType &value);
    /*!
     * \brief Start beginning of array.
     * \param multi_line whether to start an multi_line array.
     */
    void begin_array(bool multi_line = true);
    /*! \brief Finish writing an array. */
    void end_array();
    /*!
     * \brief write seperator of array, before writing next element.
     * User can proceed to call writer->write to write next item
     */
    void write_array_seperator();
    /*!
     * \brief write seperator of item, before writing next item
     */
    void write_item_seperator();
    /*!
     * \brief write value into array.
     * \param value The value of to be written.
     * \tparam ValueType The value type to be written.
     */
    template <typename ValueType>
    inline void write_array_item(const ValueType &value);

    void write_key(const std::string &key);

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
    void write_seperator();
};

class SC_INTERNAL_API json_reader {
public:
    size_t recursion_limit_;
    /*!
     * \brief Constructor.
     * \param is the input source.
     * \param recursion_limit the recursion limit of nested arrays/objects
     */
    explicit json_reader(std::istream *is, size_t recursion_limit = 64)
        : recursion_limit_(recursion_limit)
        , is_(is)
        , line_count_r_(0)
        , line_count_n_(0) {}
    /*!
     * \brief Parse next JSON string.
     */
    std::string read_string();
    /*!
     * \brief read Number.
     * \tparam ValueType type of the number
     */
    template <typename ValueType>
    ValueType read_number();
    /*
     * \brief read a bool value
     */
    bool read_bool();
    /*!
     * \brief Begin parsing an object.
     */
    void begin_object();
    /*!
     * \brief Begin parsing an array.
     */
    void begin_array();
    /*!
     * \brief Try to move to next object item.
     *  If this call is successful, user can proceed to call
     *  reader->read to read in the value.
     * \param out_key the key to the next object.
     * \return true if the read is successful, false if we are at end of the
     * object.
     */
    bool next_object_item(std::string *out_key);
    /*!
     * \brief Try to read the next element in the array.
     *  If this call is successful, user can proceed to call
     *  reader->read to read in the value.
     * \return true if the read is successful, false if we are at end of the
     * array.
     */
    bool next_array_item();
    /*!
     * \brief read next ValueType.
     * \param out_value any STL or json readable type to be read
     * \tparam ValueType the data type to be read.
     */
    template <typename ValueType>
    inline void read(ValueType *out_value);

    /*!
     * \brief read next ValueType.
     * \param out_value any STL or json readable type to be read
     */
    template <typename ValueType>
    inline ValueType read();
    /*!
     * \brief read just before next nonspace but not read that.
     * \return the next nonspace character.
     */
    int peeknext_nonspace();

    void expect_next_object_key(const std::string &expected_key);

    void expect_object_ends();

    std::string get_error_position_str() const;
    void expect(bool cond, const char *msg) const;

private:
    /*! \brief internal reader stream */
    std::istream *is_;
    /*! \brief "\\r" counter */
    size_t line_count_r_;
    /*! \brief "\\n" counter */
    size_t line_count_n_;
    // position in the line
    size_t line_position_ = 0;
    /*!
     * \brief record how many element processed in
     *  current array/object scope.
     */
    std::vector<size_t> scope_counter_;
    // reads a character and check if it is `ch`
    void read_and_expect(char ch);

    /*!
     * \brief read next nonspace character.
     * \return the next nonspace character.
     */
    int next_nonspace();
    // /*!
    //  * \brief read just before next nonspace but not read that.
    //  * \return the next nonspace character.
    //  */
    // inline int peeknext_nonspace();
    /*!
     * \brief Takes the next char from the input source.
     * \return the next character.
     */
    int next_char();
    /*!
     * \brief Returns the next char from the input source.
     * \return the next character.
     */
    int peeknext_char();
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
    inline static ValueType read(json_reader *reader) {
        return static_cast<ValueType>(reader->read_number<int32_t>());
    }
};

template <typename ValueType>
struct numeric_handler {
    inline static void write(json_writer *writer, const ValueType &value) {
        writer->write_number<ValueType>(value);
    }
    inline static ValueType read(json_reader *reader) {
        return reader->read_number<ValueType>();
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
    inline static TReal read(json_reader *reader) {
        typedef typename if_else_type<std::is_enum<TReal>::value,
                enum_handler<TReal>, handler<TReal>>::Type t_handler;
        return t_handler::read(reader);
    }
};

template <typename ValueType>
struct common_handler<std::shared_ptr<ValueType>> {
    inline static void write(
            json_writer *writer, const std::shared_ptr<ValueType> &value) {
        writer->write(*value);
    }

    inline static std::shared_ptr<ValueType> read(json_reader *reader) {
        auto ret = std::make_shared<ValueType>();
        *ret = reader->read<ValueType>();
        return ret;
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
    inline static std::pair<K, V> read(json_reader *reader) {
        reader->begin_array();
        reader->expect(reader->next_array_item(), "Expect array of length 2");
        std::pair<K, V> kv;
        kv.first = handler<K>::read(reader);
        reader->expect(reader->next_array_item(), "Expect array of length 2");
        kv.second = handler<V>::read(reader);
        reader->expect(!reader->next_array_item(), "Expect array of length 2");
        return kv;
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
    inline static ContainerType read(json_reader *reader) {
        ContainerType array;
        reader->begin_array();
        while (reader->next_array_item()) {
            array.insert(array.end(), handler<ElemType>::read(reader));
        }
        return array;
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
    inline static ContainerType read(json_reader *reader) {
        typedef typename ContainerType::mapped_type ElemType;
        ContainerType map;
        reader->begin_object();
        std::string key;
        while (reader->next_object_item(&key)) {
            map.insert(
                    std::make_pair(std::move(key), reader->read<ElemType>()));
        }
        return map;
    }
};

template <>
struct handler<std::string> {
    inline static void write(json_writer *writer, const std::string &value) {
        writer->write_string(value);
    }
    inline static std::string read(json_reader *reader) {
        return reader->read_string();
    }
};

template <>
struct handler<bool> {
    inline static void write(json_writer *writer, const bool &value) {
        writer->write_bool(value);
    }
    inline static bool read(json_reader *reader) { return reader->read_bool(); }
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
    inline static T read(json_reader *reader) {
        return t_handler::read(reader);
    }
};

template <>
struct SC_INTERNAL_API handler<any_t> {
    static void write(json_writer *writer, const any_t &data);
    static any_t read(json_reader *reader);
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
    static reflection::general_object_t read(json_reader *reader);
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

template <typename ValueType>
inline void json_writer::write_number(const ValueType &v) {
    *os_ << v;
}

template <typename ValueType>
inline void json_writer::write_keyvalue(
        const std::string &key, const ValueType &value) {
    if (skip_temp_ && key.find("temp.") == 0) { return; }
    write_key(key);
    json::handler<ValueType>::write(this, value);
}

template <typename ValueType>
inline void json_writer::write_array_item(const ValueType &value) {
    this->write_array_seperator();
    json::handler<ValueType>::write(this, value);
}

template <typename ValueType>
inline void json_writer::write(const ValueType &value) {
    size_t nscope = scope_multi_line_.size();
    handler<ValueType>::write(this, value);
    assert(nscope == scope_multi_line_.size());
}

template <typename ValueType>
inline ValueType json_reader::read_number() {
    ValueType out_value;
    auto old_cnt = is_->tellg();
    *is_ >> out_value;
    expect(!is_->fail(), "Bad value");
    auto new_cnt = is_->tellg();
    if (new_cnt == -1) new_cnt = old_cnt + std::streamoff(1);
    line_position_ += (new_cnt - old_cnt);
    return out_value;
}

template <>
inline void json_reader::read(reflection::general_ref_t *out_value) {
    json::handler<reflection::general_ref_t>::read(this, out_value);
}

template <typename T>
inline void json_reader::read(T *out_value) {
    *out_value = json::handler<T>::read(this);
}

template <typename T>
inline T json_reader::read() {
    return json::handler<T>::read(this);
}

} // namespace json
} // namespace sc

#endif
