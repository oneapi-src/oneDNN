
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

#include <limits>
#include "any_map.hpp"
#include "any_reflection_cvt.hpp"
#include "json.hpp"
#include "reflection.hpp"
namespace sc {
namespace json {

static void json_assert(bool cond, const char *msg) {
    if (!cond) { throw json_error(msg); }
}

void json_writer::write_key(const std::string &key) {
    if (scope_counter_.back() > 0) { *os_ << ","; }
    write_seperator();
    *os_ << '\"';
    *os_ << key;
    *os_ << "\": ";
    scope_counter_.back() += 1;
}

void json_writer::begin_object(bool multi_line) {
    *os_ << "{";
    scope_multi_line_.push_back(multi_line);
    scope_counter_.push_back(0);
}

void json_writer::write_bool(const bool &v) {
    *os_ << v;
}

void json_writer::write_string(const std::string &s) {
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

void json_writer::begin_array(bool multi_line) {
    *os_ << '[';
    scope_multi_line_.push_back(multi_line);
    scope_counter_.push_back(0);
}

void json_writer::end_array() {
    json_assert(!scope_counter_.empty() && !scope_multi_line_.empty(),
            "Unmatched end_array");

    bool newline = scope_multi_line_.back();
    size_t nelem = scope_counter_.back();
    scope_multi_line_.pop_back();
    scope_counter_.pop_back();
    if (newline && nelem != 0) write_seperator();

    *os_ << ']';
}

void json_writer::write_array_seperator() {
    if (scope_counter_.back() != 0) { *os_ << ", "; }
    scope_counter_.back() += 1;
    write_seperator();
}

void json_writer::write_item_seperator() {
    *os_ << ", ";
    write_seperator();
}

void json_writer::end_object() {
    // TODO(Zhichen): Replace comparison by log check
    json_assert(!scope_counter_.empty() && !scope_multi_line_.empty(),
            "Unmatched end_object");
    bool newline = scope_multi_line_.back();
    size_t nelem = scope_counter_.back();
    scope_multi_line_.pop_back();
    scope_counter_.pop_back();
    if (newline && nelem != 0) write_seperator();
    *os_ << '}';
}

void json_writer::write_seperator() {
    if (pretty_print_
            && (scope_multi_line_.empty() || scope_multi_line_.back())) {
        *os_ << '\n';
        *os_ << std::string(scope_multi_line_.size() * 2, ' ');
    }
}

std::string json_reader::get_error_position_str() const {
    std::stringstream ss;
    ss << "json reader error at line " << line_count_n_ + 1 << ':'
       << line_position_ << " - ";
    return ss.str();
}

void json_reader::read_and_expect(char ch) {
    if (next_nonspace() != ch) {
        std::stringstream ss;
        ss << get_error_position_str();
        ss << "Expecting \'" << ch << '\'';
        throw json_error(ss.str());
    }
}

void json_reader::expect(bool cond, const char *msg) const {
    if (!cond) { throw json_error(get_error_position_str() + msg); }
}

int json_reader::next_char() {
    line_position_++;
    return is_->get();
}

int json_reader::peeknext_char() {
    return is_->peek();
}

int json_reader::next_nonspace() {
    int ch;
    do {
        ch = next_char();
        if (ch == '\n') {
            ++line_count_n_;
            line_position_ = 0;
        }
        if (ch == '\r') ++line_count_r_;
    } while (isspace(ch));
    return ch;
}

int json_reader::peeknext_nonspace() {
    int ch;
    while (true) {
        ch = peeknext_char();
        if (ch == '\n') {
            ++line_count_n_;
            line_position_ = 0;
        }
        if (ch == '\r') ++line_count_r_;
        if (!isspace(ch)) break;
        next_char();
    }
    return ch;
}

std::string json_reader::read_string() {
    read_and_expect('\"');
    std::string output;
    while (true) {
        int ch = next_char();
        if (ch == '\\') {
            char sch = static_cast<char>(next_char());
            switch (sch) {
                case 'r': output += '\r'; break;
                case 'n': output += '\n'; break;
                case '\\': output += '\\'; break;
                case 't': output += '\t'; break;
                case '\"': output += '\"'; break;
                default: expect(false, "Unknown string escape character");
            }
        } else {
            if (ch == '\"') break;
            output += static_cast<char>(ch);
        }
        expect(ch != EOF && ch != '\r' && ch != '\n',
                "Bad character in string");
    }
    return output;
}

bool json_reader::read_bool() {
    bool out_value;
    auto old_cnt = is_->tellg();
    *is_ >> out_value;
    expect(!is_->fail(), "Bad value");
    auto new_cnt = is_->tellg();
    if (new_cnt == -1) new_cnt = old_cnt + std::streamoff(1);
    line_position_ += (new_cnt - old_cnt);
    return out_value;
}

void json_reader::begin_object() {
    read_and_expect('{');
    scope_counter_.push_back(0);
    expect(scope_counter_.size() < recursion_limit_, "Recursion limit reached");
}

void json_reader::begin_array() {
    read_and_expect('[');
    scope_counter_.push_back(0);
    expect(scope_counter_.size() < recursion_limit_, "Recursion limit reached");
}

bool json_reader::next_object_item(std::string *out_key) {
    bool next = true;
    expect(!scope_counter_.empty(),
            "next_object_item should be called after begin_object");
    if (scope_counter_.back() != 0) {
        int ch = next_nonspace();
        if (ch == '}') {
            next = false;
        } else {
            expect(ch == ',', "Expecting ,");
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
        *out_key = read_string();
        read_and_expect(':');
        return true;
    }
}

bool json_reader::next_array_item() {
    bool next = true;
    expect(!scope_counter_.empty(),
            "next_array_item should be called after begin_array");
    if (scope_counter_.back() != 0) {
        int ch = next_nonspace();
        if (ch == ']') {
            next = false;
        } else {
            expect(ch == ',', "Expecting ,");
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

#define JSON_ASSERT(reader, cond, ...) \
    if (!(cond)) { \
        ::std::stringstream ss; \
        ss << reader->get_error_position_str(); \
        ss << __VA_ARGS__ << "\n"; \
        throw json_error(ss.str()); \
    }

void json_reader::expect_next_object_key(const std::string &expected_key) {
    std::string key;
    bool has_next = next_object_item(&key);
    JSON_ASSERT(this, has_next && key == expected_key,
            "Expecting key=" << expected_key);
}

void json_reader::expect_object_ends() {
    std::string key;
    bool has_next = next_object_item(&key);
    expect(!has_next, "Expecting end of object");
}

void handler<any_t>::write(json_writer *writer, const any_t &data) {
    auto gen_ref = utils::any_to_general_ref(data);
    writer->begin_object(false);
    writer->write_keyvalue("type", gen_ref.type_.to_string());
    writer->write_keyvalue("data", gen_ref);
    writer->end_object();
}

any_t handler<any_t>::read(json_reader *reader) {
    reader->begin_object();
    std::string key, typen;
    bool hasnext = reader->next_object_item(&key);
    reader->expect(hasnext, "Reading empty JSON object for any");
    reader->expect(key == "type", "Expecting key=\"type\" for any");
    reader->read(&typen);

    auto ty = reflection::get_type_by_name(typen);
    JSON_ASSERT(reader, ty, "Cannot find reflection type: " << typen);
    any_t ret = any_t::make_by_type(ty);
    reflection::general_ref_t ref {ret.get_raw(), *ty};
    reader->expect_next_object_key("data");
    reader->read(&ref);

    hasnext = reader->next_object_item(&key);
    reader->expect(!hasnext, "Junk data after key=\"data\" for any");

    return ret;
}
using stdanymap = std::unordered_map<std::string, any_t>;
static reflection::type general_obj_type
        = reflection::type_registry<reflection::general_object_t>::type_;
static reflection::type general_shared_obj_type
        = reflection::type_registry<reflection::shared_general_object_t>::type_;
static reflection::type std_anymap_type
        = reflection::type_registry<stdanymap>::type_;

static void write_object(json_writer *writer, std::string *class_name,
        void *data, reflection::class_metadata *meta) {
    writer->begin_object(meta->fields_.size() > 1);
    if (class_name) writer->write_keyvalue("class", *class_name);
    for (auto &f : meta->fields_) {
        reflection::general_ref_t ref {f->addresser_->get(data), f->type_};
        writer->write_keyvalue(f->name_, ref);
    }
    writer->end_object();
}

// clang-format off
// NOLINTNEXTLINE
#define PUT_VALUE(PREFIX, TYPE) virtual bool visit(TYPE *v, TYPE *v2) override { \
        writer->write(std::string(PREFIX) + std::to_string(*v)); \
        return true; \
    }

// clang-format on
using reflection::general_ref_t;
struct jsonwrite_visitor_t : public reflection::visitor_t {
    json_writer *writer;
    jsonwrite_visitor_t(json_writer *writer) : writer(writer) {}
    bool dispatch(general_ref_t *v1, general_ref_t *v2) override {
        if (v1->type_ == std_anymap_type) {
            auto anymapptr = reinterpret_cast<stdanymap *>(v1->data_);
            handler<stdanymap>::write(writer, *anymapptr);
        } else if (v1->type_ == general_obj_type) {
            auto &obj = *reinterpret_cast<reflection::general_object_t *>(
                    v1->data_);
            write_object(writer, &obj.vtable_->name_, obj.data_.get(),
                    obj.vtable_.get());
        } else if (v1->type_ == general_shared_obj_type) {
            auto &obj
                    = *reinterpret_cast<reflection::shared_general_object_t *>(
                            v1->data_);
            write_object(writer, &obj.vtable_->name_, obj.data_.get(),
                    obj.vtable_.get());
        } else {
            reflection::visitor_t::dispatch(v1, v2);
        }
        return true;
    }
    bool visit_class(general_ref_t *v1, general_ref_t *v2) override {
        write_object(writer, nullptr, v1->data_, v1->type_.meta_);
        return true;
    }
    bool visit_array(general_ref_t *v, general_ref_t *v2,
            reflection::vector_metadata *vec_meta, size_t len, size_t len2,
            size_t objsize, char *base1, char *base2) override {
        writer->begin_array();
        for (unsigned i = 0; i < len; i++) {
            reflection::general_ref_t elem {base1, vec_meta->element_type_};
            writer->write_array_item(elem);
            base1 += objsize;
        }
        writer->end_array();
        return true;
    }

    PUT_VALUE("i", int32_t)
    PUT_VALUE("I", int64_t)
    PUT_VALUE("u", uint32_t)
    PUT_VALUE("U", uint64_t)
    PUT_VALUE("b", bool)

// clang-format off
// NOLINTNEXTLINE
#define PUT_FLOAT_VALUE(PREFIX, TYPE) virtual bool visit(TYPE *v, TYPE *v2) override { \
        std::stringstream ofs; \
        ofs.precision(std::numeric_limits<TYPE>::max_digits10); \
        ofs << (PREFIX) << *v; \
        writer->write(ofs.str()); \
        return true; \
    }
    // clang-format on
    PUT_FLOAT_VALUE("f", float)
    PUT_FLOAT_VALUE("d", double)

    bool visit(std::string *v, std::string *v2) override {
        writer->write(std::string("s") + *v);
        return true;
    }
};
#undef PUT_VALUE

void handler<reflection::general_ref_t>::write(
        json_writer *writer, const reflection::general_ref_t &data) {
    if (data.type_.meta_ && data.type_.meta_->vtable_
            && data.type_.meta_->vtable_->json_serailize_) {
        data.type_.meta_->vtable_->json_serailize_(data.data_, writer);
        return;
    }
    jsonwrite_visitor_t v {writer};
    v.dispatch(const_cast<general_ref_t *>(&data), nullptr);
}

void handler<reflection::general_object_t>::write(
        json_writer *writer, const reflection::general_object_t &data) {
    auto ret = reflection::general_ref_t::from(data);
    writer->write(ret);
}

static void read_object_body(
        json_reader *reader, reflection::general_ref_t *data) {
    std::string key;
    auto meta = data->type_.meta_;
    std::unordered_set<void *> met;
    while (reader->next_object_item(&key)) {
        auto itr = meta->field_map_.find(key);
        JSON_ASSERT(reader, itr != meta->field_map_.end(),
                "Cannot find field " << key << " in class " << meta->name_);
        auto &fieldmeta = itr->second;
        JSON_ASSERT(reader, met.count(fieldmeta) == 0,
                "The field " << key << " occurs more than once");
        met.insert(fieldmeta);
        reflection::general_ref_t ref {
                fieldmeta->addresser_->get(data->data_), fieldmeta->type_};
        reader->read(&ref);
    }
    reader->expect(
            met.size() == meta->field_map_.size(), "Missing fields in JSON");
    if (meta->initalizer_) { meta->initalizer_(data->data_); }
}

static reflection::general_object_t read_tagged_object(json_reader *reader) {
    reader->begin_object();
    std::string key, classname;
    bool hasnext = reader->next_object_item(&key);
    reader->expect(hasnext,
            "Reading empty JSON object for reflection::general_object_t");
    reader->expect(key == "class",
            "The first entry of a tagged object must have the key \"class\"");
    reader->read(&classname);
    auto meta = reflection::get_metadata(classname);
    JSON_ASSERT(reader, meta, "Cannot find class name: " << classname);
    reflection::general_object_t ret = meta->make_instance();
    auto ref = reflection::general_ref_t::from(ret);
    read_object_body(reader, &ref);
    return ret;
}

void handler<reflection::general_ref_t>::read(
        json_reader *reader, reflection::general_ref_t *data) {
    if (data->type_.meta_ && data->type_.meta_->vtable_
            && data->type_.meta_->vtable_->json_deserailize_) {
        data->type_.meta_->vtable_->json_deserailize_(data->data_, reader);
        return;
    }

    if (data->type_ == std_anymap_type) {
        auto anymapptr = reinterpret_cast<stdanymap *>(data->data_);
        *anymapptr = handler<stdanymap>::read(reader);
    } else if (data->type_ == general_obj_type) {
        auto &obj = *reinterpret_cast<reflection::general_object_t *>(
                data->data_);
        obj = read_tagged_object(reader);
    } else if (data->type_ == general_shared_obj_type) {
        auto &target = *reinterpret_cast<reflection::shared_general_object_t *>(
                data->data_);
        reflection::general_object_t obj = read_tagged_object(reader);
        target = std::move(obj);
    } else if (data->type_.array_depth_ >= 1) {
        reader->expect(data->type_.meta_
                        && data->type_.meta_->vector_kind_
                                != reflection::vector_kind::not_vector,
                "Bad reflection metadata");
        auto meta
                = static_cast<reflection::vector_metadata *>(data->type_.meta_);
        reflection::type elem_type = meta->element_type_;
        JSON_ASSERT(reader,
                meta->vector_kind_ == reflection::vector_kind::std_array
                        || meta->size(data->data_) == 0,
                "Expecting std::array or an empty vector");
        size_t cnt = 0;
        reader->begin_array();
        while (reader->next_array_item()) {
            if (meta->vector_kind_ == reflection::vector_kind::std_array) {
                JSON_ASSERT(reader, cnt < meta->size(data->data_),
                        "Too many values inserted to const sized array");
            } else {
                meta->push_empty(data->data_);
            }
            auto ptr_elem = meta->ptr_of_element(data->data_, cnt);
            reflection::general_ref_t elem {ptr_elem, elem_type};
            read(reader, &elem);
            cnt += 1;
        }
    } else if (data->type_.base_ == reflection::basic_type::t_class) {
        reader->begin_object();
        read_object_body(reader, data);
    } else {
        std::string raw = reader->read_string();
        reader->expect(!raw.empty(), "Empty string for general ref");
        std::string rawdata = raw.substr(1);
#define PUT_VALUE(T, CVT) \
    { \
        auto ty = reflection::type_registry<T>::type_; \
        JSON_ASSERT(reader, data->type_ == ty, \
                "Got " #T " in JSON, expecting " << data->type_.to_string()); \
        *(T *)data->data_ = T(std::CVT(rawdata)); \
        break; \
    }

        switch (raw[0]) {
            case 'i': PUT_VALUE(int32_t, stoi)
            case 'u': PUT_VALUE(uint32_t, stoul)
            case 'I': PUT_VALUE(int64_t, stoll)
            case 'U': PUT_VALUE(uint64_t, stoull)
            case 'f': PUT_VALUE(float, stof)
            case 'd': PUT_VALUE(double, stod)
            case 'b': PUT_VALUE(bool, stoul)
            case 's': {
                auto ty = reflection::type_registry<std::string>::type_;
                JSON_ASSERT(reader, data->type_ == ty,
                        "Got string in JSON, expecting "
                                << data->type_.to_string());
                *(std::string *)data->data_ = rawdata;
                break;
            }
            default: {
                throw json_error(reader->get_error_position_str()
                        + "bad data tag for general ref: " + raw);
                break;
            }
        }
#undef PUT_VALUE
    }
} // namespace json

reflection::general_object_t handler<reflection::general_object_t>::read(
        json_reader *reader) {
    return read_tagged_object(reader);
}

} // namespace json

} // namespace sc
