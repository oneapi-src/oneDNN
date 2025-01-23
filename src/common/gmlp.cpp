#include <oneapi/dnnl/experimental/dnnl_experimental.h>

#include "opdesc.hpp"
#include "primitive_desc_iface.hpp"
#include "gated_mlp_pd.hpp"
#include "gated_mlp_utils.hpp"

dnnl_status_t dnnl_gmlp_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc_iface, dnnl_engine_t engine,
        const_dnnl_memory_desc_t v, const_dnnl_memory_desc_t wg,
        const_dnnl_memory_desc_t wu, const_dnnl_memory_desc_t wd,
        const_dnnl_memory_desc_t dst, const_dnnl_primitive_attr_t attr,
        const_dnnl_primitive_attr_t gate_attr, const_dnnl_primitive_attr_t up_attr,
        const_dnnl_primitive_attr_t down_attr) {
    dnnl::impl::gated_mlp_desc_t gated_mlp_desc = dnnl::impl::create_gated_mlp_desc(v,
            wg, wu, wd, dst, gate_attr, up_attr, down_attr);
    return dnnl::impl::primitive_desc_create(primitive_desc_iface, engine,
            (const dnnl::impl::op_desc_t *)&gated_mlp_desc, nullptr, attr);
}
