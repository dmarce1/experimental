/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <cosmictiger/cuda/cuda_runtime.hpp>
#include <cosmictiger/profiler.hpp>

#include <cosmictiger/lock_guards.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <hpx/async_cuda/cuda_future.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/runtime_local/service_executors.hpp>
#include <hpx/runtime_local/shutdown_function.hpp>

#include <exception>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <utility>

namespace cosmictiger::cuda {

namespace detail {

    void checkStatus(cudaError_t status, char const* operation) {
        PROFILE();
        if (status != cudaSuccess) {
            // Clear this I/O worker's sticky launch status after recording the
            // operation's error so an unrelated later submission cannot inherit it.
            (void) cudaGetLastError();
            if (status == cudaErrorMemoryAllocation) {
                throw std::bad_alloc();
            }
            throw CudaError(status, operation);
        }
    }

    class Stream {
    public:
        explicit Stream(int deviceId)
          : deviceId_(deviceId) {
            PROFILE();
            checkStatus(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        }

        ~Stream() {
            PROFILE();
            (void) cudaSetDevice(deviceId_);
            (void) cudaStreamDestroy(stream_);
        }

        cudaStream_t handle() const noexcept {
            PROFILE();
            return stream_;
        }

    private:
        int deviceId_;
        cudaStream_t stream_ = nullptr;
    };

    class StreamRegistry {
    public:
        void registerCleanup() {
            PROFILE();
            bool registerHook = false;
            {
                LockGuard guard(mutex_);
                if (!cleanupRegistered_) {
                    cleanupRegistered_ = true;
                    registerHook = true;
                }
            }
            if (registerHook) {
                try {
                    hpx::register_shutdown_function(hpx::annotated_function(
                        [this] {
                            PROFILE();
                            clear();
                        },
                        "cuda::StreamRegistry cleanup"));
                } catch (...) {
                    {
                        LockGuard guard(mutex_);
                        cleanupRegistered_ = false;
                    }
                    throw;
                }
            }
        }

        std::shared_ptr<Stream> get(int deviceId) {
            PROFILE();
            {
                LockGuard guard(mutex_);
                auto const found = streams_.find(deviceId);
                if (found != streams_.end()) {
                    return found->second;
                }
            }
            auto candidate = std::make_shared<Stream>(deviceId);
            std::shared_ptr<Stream> selected;
            {
                LockGuard guard(mutex_);
                auto const [entry, inserted] = streams_.try_emplace(deviceId, candidate);
                selected = entry->second;
            }
            return selected;
        }

    private:
        void clear() {
            PROFILE();
            std::map<int, std::shared_ptr<Stream>> retired;
            {
                LockGuard guard(mutex_);
                retired.swap(streams_);
                cleanupRegistered_ = false;
            }
            // Destroy streams outside the lock; outstanding completions retain
            // their own owners until the associated GPU work has completed.
        }

        hpx::spinlock mutex_;
        std::map<int, std::shared_ptr<Stream>> streams_;
        bool cleanupRegistered_ = false;
    };

    StreamRegistry& streamRegistry() {
        PROFILE();
        static StreamRegistry registry;
        return registry;
    }

    template <typename Operation>
    auto submit(Operation operation, bool usesStream = false) -> hpx::future<decltype(operation())> {
        using Result = decltype(operation());
        PROFILE();
        try {
            if (usesStream) {
                streamRegistry().registerCleanup();
            }
            hpx::execution::experimental::io_pool_executor executor;
            return hpx::parallel::execution::async_execute(executor, hpx::annotated_function(std::move(operation), "cuda::submit"));
        } catch (...) {
            return hpx::make_exceptional_future<Result>(std::current_exception());
        }
    }

    template <typename Operation>
    auto submitOnDevice(int deviceId, Operation operation, bool usesStream = false) -> hpx::future<decltype(operation())> {
        PROFILE();
        return submit(hpx::annotated_function(
                          [deviceId, operation = std::move(operation)]() mutable {
                              PROFILE();
                              checkStatus(cudaSetDevice(deviceId), "cudaSetDevice");
                              return operation();
                          },
                          "cuda::submitOnDevice"),
            usesStream);
    }

    hpx::future<void> completion(
        cudaStream_t stream, std::shared_ptr<Stream> ownedStream = {}, hpx::move_only_function<void(cudaStream_t)> retainedOperation = {}) {
        PROFILE();
        // CUDA invokes HPX's callback on a CUDA runtime thread. Bridge through an
        // asynchronous HPX continuation so user continuations and destructors can
        // safely submit more CUDA operations when the public future becomes ready.
        std::shared_ptr<hpx::move_only_function<void(cudaStream_t)>> retained;
        try {
            retained = std::make_shared<hpx::move_only_function<void(cudaStream_t)>>(std::move(retainedOperation));
            return hpx::cuda::experimental::detail::get_future_with_callback(stream).then(hpx::launch::async,
                hpx::annotated_function(
                    [ownedStream, retained](hpx::future<void> completed) {
                        PROFILE();
                        completed.get();
                    },
                    "cuda::completion"));
        } catch (...) {
            // Preserve capture and stream lifetimes if constructing a completion
            // future fails after work has already been enqueued. Only this error
            // path blocks the I/O worker; normal completions use HPX callbacks.
            (void) cudaStreamSynchronize(stream);
            throw;
        }
    }

}    // namespace detail

CudaError::CudaError(cudaError_t status, char const* operation)
  : std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status))
  , code_(status) {
    PROFILE();
}

cudaError_t CudaError::code() const noexcept {
    PROFILE();
    return code_;
}

hpx::future<int> deviceCount() {
    PROFILE();
    return detail::submit(hpx::annotated_function(
        [] {
            PROFILE();
            int count = 0;
            detail::checkStatus(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
            return count;
        },
        "cuda::deviceCount"));
}

hpx::future<int> device() {
    PROFILE();
    return detail::submitOnDevice(0,
        hpx::annotated_function(
            [] {
                PROFILE();
                int deviceId = 0;
                detail::checkStatus(cudaGetDevice(&deviceId), "cudaGetDevice");
                return deviceId;
            },
            "cuda::device"));
}

hpx::future<int> deviceAttribute(cudaDeviceAttr attribute, int deviceId) {
    PROFILE();
    return detail::submit(hpx::annotated_function(
        [attribute, deviceId] {
            PROFILE();
            int value = 0;
            detail::checkStatus(cudaDeviceGetAttribute(&value, attribute, deviceId), "cudaDeviceGetAttribute");
            return value;
        },
        "cuda::deviceAttribute"));
}

hpx::future<cudaDeviceProp> deviceProperties(int deviceId) {
    PROFILE();
    return detail::submit(hpx::annotated_function(
        [deviceId] {
            PROFILE();
            cudaDeviceProp properties{};
            detail::checkStatus(cudaGetDeviceProperties(&properties, deviceId), "cudaGetDeviceProperties");
            return properties;
        },
        "cuda::deviceProperties"));
}

hpx::future<cudaPointerAttributes> pointerAttributes(void const* pointer) {
    PROFILE();
    return detail::submit(hpx::annotated_function(
        [pointer] {
            PROFILE();
            cudaPointerAttributes attributes{};
            detail::checkStatus(cudaPointerGetAttributes(&attributes, pointer), "cudaPointerGetAttributes");
            return attributes;
        },
        "cuda::pointerAttributes"));
}

hpx::future<void*> malloc(size_t bytes, int deviceId) {
    PROFILE();
    return detail::submitOnDevice(deviceId,
        hpx::annotated_function(
            [bytes] {
                PROFILE();
                void* pointer = nullptr;
                detail::checkStatus(cudaMalloc(&pointer, bytes), "cudaMalloc");
                return pointer;
            },
            "cuda::malloc"));
}

hpx::future<void*> mallocManaged(size_t bytes, unsigned flags, int deviceId) {
    PROFILE();
    return detail::submitOnDevice(deviceId,
        hpx::annotated_function(
            [bytes, flags] {
                PROFILE();
                void* pointer = nullptr;
                detail::checkStatus(cudaMallocManaged(&pointer, bytes, flags), "cudaMallocManaged");
                return pointer;
            },
            "cuda::mallocManaged"));
}

hpx::future<void> free(void* pointer, int deviceId) {
    PROFILE();
    if (pointer == nullptr) {
        return hpx::make_ready_future();
    }
    return detail::submitOnDevice(deviceId,
        hpx::annotated_function(
            [pointer] {
                PROFILE();
                detail::checkStatus(cudaFree(pointer), "cudaFree");
            },
            "cuda::free"));
}

hpx::future<cudaStream_t> streamCreate(unsigned flags, int deviceId) {
    PROFILE();
    return detail::submitOnDevice(deviceId,
        hpx::annotated_function(
            [flags] {
                PROFILE();
                cudaStream_t stream = nullptr;
                detail::checkStatus(cudaStreamCreateWithFlags(&stream, flags), "cudaStreamCreateWithFlags");
                return stream;
            },
            "cuda::streamCreate"));
}

hpx::future<void> streamDestroy(cudaStream_t stream, int deviceId) {
    PROFILE();
    if (stream == nullptr) {
        return hpx::make_exceptional_future<void>(std::invalid_argument("Cannot destroy the default CUDA interface stream"));
    }
    return synchronize(stream, deviceId)
        .then(hpx::launch::async,
            hpx::annotated_function(
                [stream, deviceId](hpx::future<void> completed) {
                    PROFILE();
                    completed.get();
                    return detail::submitOnDevice(deviceId,
                        hpx::annotated_function(
                            [stream] {
                                PROFILE();
                                detail::checkStatus(cudaStreamDestroy(stream), "cudaStreamDestroy");
                            },
                            "cuda::streamDestroy"));
                },
                "cuda::streamDestroy completion"));
}

hpx::future<void> memcpy(void* destination, void const* source, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream, int deviceId) {
    PROFILE();
    return hpx::future<void>(detail::submitOnDevice(deviceId,
        hpx::annotated_function(
            [destination, source, bytes, kind, stream, deviceId] {
                PROFILE();
                auto ownedStream = stream == nullptr ? detail::streamRegistry().get(deviceId) : std::shared_ptr<detail::Stream>{};
                cudaStream_t const selectedStream = ownedStream ? ownedStream->handle() : stream;
                detail::checkStatus(cudaMemcpyAsync(destination, source, bytes, kind, selectedStream), "cudaMemcpyAsync");
                return detail::completion(selectedStream, std::move(ownedStream));
            },
            "cuda::memcpy"),
        stream == nullptr));
}

hpx::future<void> prefetch(std::vector<ManagedMemoryRange> ranges, int destination, cudaStream_t stream, int deviceId) {
    PROFILE();
    return hpx::future<void>(detail::submitOnDevice(deviceId,
        hpx::annotated_function(
            [ranges = std::move(ranges), destination, stream, deviceId] {
                PROFILE();
                auto ownedStream = stream == nullptr ? detail::streamRegistry().get(deviceId) : std::shared_ptr<detail::Stream>{};
                cudaStream_t const selectedStream = ownedStream ? ownedStream->handle() : stream;
                try {
                    for (auto const& range : ranges) {
                        if (range.bytes != 0) {
                            detail::checkStatus(cudaMemPrefetchAsync(range.pointer, range.bytes, destination, selectedStream), "cudaMemPrefetchAsync");
                        }
                    }
                } catch (...) {
                    // Earlier ranges may already be migrating. Complete them
                    // before the caller can release buffers after a failure.
                    (void) cudaStreamSynchronize(selectedStream);
                    throw;
                }
                return detail::completion(selectedStream, std::move(ownedStream));
            },
            "cuda::prefetch submission"),
        stream == nullptr));
}

hpx::future<void> launch(hpx::move_only_function<void(cudaStream_t)> operation, cudaStream_t stream, int deviceId) {
    PROFILE();
    return hpx::future<void>(detail::submitOnDevice(deviceId,
        hpx::annotated_function(
            [operation = std::move(operation), stream, deviceId]() mutable {
                PROFILE();
                if (operation.empty()) {
                    throw std::invalid_argument("CUDA launch operation is empty");
                }
                hpx::move_only_function<void(cudaStream_t)> annotatedOperation(hpx::annotated_function(std::move(operation), "cuda::launch callback"));
                auto ownedStream = stream == nullptr ? detail::streamRegistry().get(deviceId) : std::shared_ptr<detail::Stream>{};
                cudaStream_t const selectedStream = ownedStream ? ownedStream->handle() : stream;
                try {
                    annotatedOperation(selectedStream);
                    detail::checkStatus(cudaGetLastError(), "CUDA kernel launch");
                } catch (...) {
                    // A callable can enqueue more than one kernel before a later launch
                    // fails. Finish that work before destroying its retained captures.
                    (void) cudaStreamSynchronize(selectedStream);
                    throw;
                }
                return detail::completion(selectedStream, std::move(ownedStream), std::move(annotatedOperation));
            },
            "cuda::launch submission"),
        stream == nullptr));
}

hpx::future<void> synchronize(cudaStream_t stream, int deviceId) {
    PROFILE();
    return hpx::future<void>(detail::submitOnDevice(deviceId,
        hpx::annotated_function(
            [stream, deviceId] {
                PROFILE();
                auto ownedStream = stream == nullptr ? detail::streamRegistry().get(deviceId) : std::shared_ptr<detail::Stream>{};
                cudaStream_t const selectedStream = ownedStream ? ownedStream->handle() : stream;
                return detail::completion(selectedStream, std::move(ownedStream));
            },
            "cuda::synchronize"),
        stream == nullptr));
}

}    // namespace cosmictiger::cuda
