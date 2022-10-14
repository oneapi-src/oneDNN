# Primitive Memory Caching Support for Arm Compute Library objects (RFC)

## Introduction

The goal for this RFC is to highlight the memory allocation overhead we currently incur for Arm Compute Library (ACL) objects, and propose ACL wrapper architecture changes to allow reuse of the allocated memory across the cached primitives.

The implementation is available [here](https://github.com/oneapi-src/oneDNN/compare/master...snadampal:oneDNN:acl_mem_pools) which is dependent on the [ACL cached memory manager](https://review.mlplatform.org/c/ml/ComputeLibrary/+/8416). No new test case development is required to test this feature. Benchdnn convolution tests or Tensorflow and PyTorch inference tests can be used.

### Motivation

Currently oneDNN primitive caching feature is not used for ACL because the ACL objects are non-immutable. The ACL objects own and manager the memory allocations for the tensor buffers. Since the ACL objects are treated as primitives resources and created/destroyed for every user primitive, the tensor memory is getting allocted/freed repeatedly. These dynamic allocations in the critical path are dominating the inference latencies and are not acceptable for the real use cases. Since the ACL object can not be cached, we need a mechanism to cache the memory managed by ACL and reuse across the tensors.

## Proposal

The proposal is to enable ACL memory manager for the GEMM objects, cache the memory manager as part of the primitive implementation, and reuse it across the cached primitives. An ideal solution would be to reuse the memory pool across any primitives, not just the cached ones, but today the ACL memory manager manages the memory at the tensor group granularity, hence taking this approach. The key changes required are listed below:

- Create ACL memory manager object inside the primitive class: `src/cpu/aarch64/`
- Construct ACL objects with the memory manager: `src/cpu/aarch64/`
- Add a new API, to `primitive_t` class,to destroy the resources created via `create_resource()` API: `src/common/primitive.hpp`
- Call the new `destroy_resource()` API from the `dnnl_primitive` destructor: `src/common/primitive_iface.cpp`
- Implement the `destroy_resource()` method for the ACL primitives that caches the memory manager object: `src/cpu/aarch64`

Alternate solutions considered:
1. Caching the ACL object: This breaks the identical primitive concurrency execution and hence not a viable option
2. Using scratchpad memory for tensor buffer: This will address the memory foot print issue, however this moves the whole memory management responsibility to the user/high level framework. This duplicates the implementation effort for every framework and takes longer to stabilize the feature. This approach also requires major changes to the ACL GEMM workspace management to be able to import the buffers owned by user.

Hence proposing the intermediate approach where the memory management still lies with the ACL and oneDNN, and caches only the underlying memory pools, not the whole ACL object.


### Instantiating Cached Memory Manager and constructing ACL object with it `src/cpu/aarch64/`

ACL provides a public [MemoryManager interface](https://github.com/ARM-software/ComputeLibrary/blob/master/arm_compute/runtime/IMemoryManager.h) that is responsible for managing the life cycle of the memory allocations required for ACL tensors. Curently this interface supports only the OnDemand allocations, and has been implemented by `OnDemandMemoryManager`. The Memory Manager interface has been extended to allow reserving and caching memory pools beyond the ACL object life cycle. The extended interface has been implemented as `CachedMemoryManager`.

The CacheMemeoryManager definition in ACL is as below:

```c++
class MemoryManagerCached : public IMemoryManager
{
public:
    /** Default Constructor */
    MemoryManagerCached(std::shared_ptr<ILifetimeManager> lifetime_manager, std::shared_ptr<IPoolManager> pool_manager);
    /** Prevent instances of this class to be copy constructed */
    MemoryManagerCached(const MemoryManagerCached &) = delete;
    /** Prevent instances of this class to be copied */
    MemoryManagerCached &operator=(const MemoryManagerCached &) = delete;
    /** Allow instances of this class to be move constructed */
    MemoryManagerCached(MemoryManagerCached &&) = default;
    /** Allow instances of this class to be moved */
    MemoryManagerCached &operator=(MemoryManagerCached &&) = default;

    // Inherited methods overridden:
    ILifetimeManager *lifetime_manager() override;
    IPoolManager     *pool_manager() override;
    void populate(IAllocator &allocator, size_t num_pools) override;
    void clear() override;
    void reserve_pools(IAllocator &allocator, size_t num_pools) override;
    void unreserve_pools(size_t num_pools) override;

private:
    std::shared_ptr<ILifetimeManager> _lifetime_mgr; /**< Lifetime manager */
    std::shared_ptr<IPoolManager>     _pool_mgr;     /**< Memory pool manager */
};
```

The `CachedMemoryManager` object is instantiated as part of the `primitive_t` derived acl gemm object, the logic is [here](https://github.com/oneapi-src/oneDNN/compare/master...snadampal:oneDNN:acl_mem_pools#diff-692c8f2f90f2565e2eef0651fe9c38b63c60e072ea5dec320e936c23cc706953R128).

The ACL object itself is still outside the `primitive_t` class and is instantiated as part of the resource object during `create_resource` call. The main change here is to construct the ACL object with the `CachedMemoryManager` which is responsible for all the ACL memory allocations.

Cached Memory Manager creation is [here](https://github.com/oneapi-src/oneDNN/compare/master...snadampal:oneDNN:acl_mem_pools#diff-692c8f2f90f2565e2eef0651fe9c38b63c60e072ea5dec320e936c23cc706953R126)
ACL object creation is [here](https://github.com/oneapi-src/oneDNN/compare/master...snadampal:oneDNN:acl_mem_pools#diff-692c8f2f90f2565e2eef0651fe9c38b63c60e072ea5dec320e936c23cc706953R33)


### Managing the primitive resource and memory life cycle `src/common` and `src/cpu/aarch64`

Once the ACL gemm object is configured and all the tensor sizes are finalized, the `CachedMemoryManager` reserves memory pool for the current primitive. The memory pool is kept reserved as long as the `dnnl_primitive` object is alive. When the `dnnl_primitive` object is destroyed, it invokes the `destroy_resource()` API which frees the reserved pools. At ACL memory manager level, the unreserved pools are moved to the free pool list and make them available for the next ACL object constructed from the same memory manager. This cycle of `reserve_pool` and `unreserve_pool` continues everytime a cached primitive is used. Hence for the cached primitives there is no memory allocation overhead!

Since this approach holds the system memory for every cached primitive there will be an increase in the use case memory foot print. However, the max cached memory can be limited by configuring the onednn primitive cache capacity appropriately. This can be done via `ONEDNN_PRIMITIVE_CACHE_CAPACITY` or `dnnl_set_primitive_cache_capacity()` API which defines the number of primitives to be cached at any time. Once the capacity limit is hit, the LRU (Least Recently Used) primitve is evicted, and hence the associated memory manager object and all the alloated memory pools get freed.

`primitive_t` new API is [here](https://github.com/oneapi-src/oneDNN/compare/master...snadampal:oneDNN:acl_mem_pools#diff-e47831315c5de573ca4cf2ba89b494b24ccc7e30e6c43dfc2e8687c59376cebcR81)
and the `dnnl_primitive` changes are [here](https://github.com/oneapi-src/oneDNN/compare/master...snadampal:oneDNN:acl_mem_pools#diff-e8485b477332f9695909b8b0fd4ecfd25e51439392a4c8c8dd987f24d8e07936R252)

the `destroy_resource()` implementation for the acl gemm primitive is [here](https://github.com/oneapi-src/oneDNN/compare/master...snadampal:oneDNN:acl_mem_pools#diff-692c8f2f90f2565e2eef0651fe9c38b63c60e072ea5dec320e936c23cc706953R148)

The ACL implementation for the Cached Memory Manager is below:

```c++
void MemoryManagerCached::reserve_pools(arm_compute::IAllocator &allocator, size_t num_pools)
{
    ARM_COMPUTE_ERROR_ON(!_lifetime_mgr);
    ARM_COMPUTE_ERROR_ON(!_pool_mgr);
    ARM_COMPUTE_ERROR_ON_MSG(!_lifetime_mgr->are_all_finalized(), "All the objects have not been finalized!");

    //First try to reserve the pool. if it fails, go for creating new one
    int success = _pool_mgr->reserve_pool();
    if (success < 0) {
        // Create pools
        auto pool_template = _lifetime_mgr->create_pool(&allocator);
        for(int i = num_pools; i > 1; --i)
        {
                auto pool = pool_template->duplicate();
                _pool_mgr->register_pool(std::move(pool));
        }
        _pool_mgr->register_pool(std::move(pool_template));
        // Now reseve again, this time it shouldn't fail
        int retry = _pool_mgr->reserve_pool();
	if (retry < 0) // Adding this explicit if check to avoid compiler unused variable error
	{
	    ARM_COMPUTE_ERROR_ON(1);
	}
    }
}

int PoolManager::reserve_pool()
{
    // It's okay to call reserve pool without setting up any pools
    // In this case, the caller is expected to first allocate pools and
    // call reserve again
    if (_free_pools.empty()) {
	return -1; //TODO: replace this with retry error code
     }
    _sem_free_pools->wait();
    arm_compute::lock_guard<arm_compute::Mutex> lock(_mtx);
    ARM_COMPUTE_ERROR_ON_MSG(_free_pools.empty(), "Empty pool must exist as semaphore has been signalled");
    _reserved_pools.splice(std::begin(_reserved_pools), _free_pools, std::begin(_free_pools));

    // Update semaphore
    _sem_reserved_pools = std::make_unique<arm_compute::Semaphore>(_reserved_pools.size());
    return 0;
}

void MemoryManagerCached::unreserve_pools(size_t num_pools)
{
    ARM_COMPUTE_ERROR_ON(!_lifetime_mgr);
    ARM_COMPUTE_ERROR_ON(!_pool_mgr);
    ARM_COMPUTE_ERROR_ON_MSG(_pool_mgr->num_pools() == 0, "Pool manager already contains pools!");
    ARM_COMPUTE_ERROR_ON_MSG((num_pools > _pool_mgr->num_pools()), "Pool manager already contains pools!");

    //Unreserve the pools
    for(int i = num_pools; i > 0; --i)
    {
        _pool_mgr->unreserve_pool();
    }
}

void PoolManager::unreserve_pool()
{
    // MemoryManagerCached is already enforcing the preconditions
    // to not to call without having any reserved pools
    // TODO: enforce some checks if other clients don't obey thesequence
    // for now there is no harm, we just ignore it
    if (_reserved_pools.empty()) {
        return;
    }
    _sem_reserved_pools->wait();
    arm_compute::lock_guard<arm_compute::Mutex> lock(_mtx);
    ARM_COMPUTE_ERROR_ON_MSG(_reserved_pools.empty(), "reserved pool must exist as semaphore has been signalled");
    _free_pools.splice(std::begin(_free_pools), _reserved_pools, std::begin(_reserved_pools));

    // Update semaphore
    _sem_free_pools = std::make_unique<arm_compute::Semaphore>(_free_pools.size());
    return;
}
```

## Limitations

The main limitation of the implementation outlined in this RFC is the increased memory foot print. Currenlty the tensor buffer memory is tied to the ACL object lifetime, hence it gets freed when the user primitive is destroyed and will be avaible for the next one. Whereas in the proposed design, the memory is tied to the primitive implementation, `primitive_t` which is avaiable only for the cached primitives not to the newly created ones.

See below the constructor of the acl gemm primitive implementation class:

```c++
acl_gemm_convolution_fwd_t(const pd_t *apd)
        : primitive_t(apd), acl_mem_mgr_obj_(std::make_shared<arm_compute::MemoryManagerCached>(
                  std::make_shared<arm_compute::BlobLifetimeManager>(),
                  std::make_shared<arm_compute::PoolManager>())) {}

```

However, we can limit the max memory by setting the onednn primitive cache capacity. If the network has the heavy primitive reuse, the substantial increase in performance outweighs the additional memory required for the use case.

## Results/validation

oneDNN's standard benchdnn test suite and the PyTorch [torchbench](https://github.com/pytorch/benchmark), and MLPerf inference tests were used for validation.
By setting the ONEDNN_PRIMITIVE_CACHE_CAPACITY to below 100, I was able to get all the benchdnn conv unit tests passing.

## Expected performance benefits

PyTorch torchbench Resnet50 showed ~3x improvement and Bert showed ~1.5x improvement for latencies compared to the current OneDNN+ACL design.

## Limited impact

The scope of this RFC, and the PRs outlined, is limited such that:

- The new API is non intrusive and doesn't need any changes to the other platforms.
- There is no impact on non-AArch64 builds.

## Implementation plan

We propose implementing the changes using a single PR with all the above changes for all ACL-enabled primitives.

With this initial implementation in place, future PRs will focus on improvements to the memory foot print reduction.
