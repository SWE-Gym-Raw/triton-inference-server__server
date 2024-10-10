// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <memory>  // For shared_ptr
#include <unordered_map>
#include <variant>

#include "../../../common.h"
#include "../../../restricted_features.h"
#include "../../../shared_memory_manager.h"
#include "../../../tracer.h"
#include "triton/common/logging.h"
#include "triton/core/tritonserver.h"


struct TRITONSERVER_Server {};

namespace triton { namespace server { namespace python {

// base exception for all Triton error code
struct TritonError : public std::runtime_error {
  explicit TritonError(const std::string& what) : std::runtime_error(what) {}
};

// triton::core::python exceptions map 1:1 to TRITONSERVER_Error_Code.
struct UnknownError : public TritonError {
  explicit UnknownError(const std::string& what) : TritonError(what) {}
};
struct InternalError : public TritonError {
  explicit InternalError(const std::string& what) : TritonError(what) {}
};
struct NotFoundError : public TritonError {
  explicit NotFoundError(const std::string& what) : TritonError(what) {}
};
struct InvalidArgumentError : public TritonError {
  explicit InvalidArgumentError(const std::string& what) : TritonError(what) {}
};
struct UnavailableError : public TritonError {
  explicit UnavailableError(const std::string& what) : TritonError(what) {}
};
struct UnsupportedError : public TritonError {
  explicit UnsupportedError(const std::string& what) : TritonError(what) {}
};
struct AlreadyExistsError : public TritonError {
  explicit AlreadyExistsError(const std::string& what) : TritonError(what) {}
};

void
ThrowIfError(TRITONSERVER_Error* err)
{
  if (err == nullptr) {
    return;
  }
  std::shared_ptr<TRITONSERVER_Error> managed_err(
      err, TRITONSERVER_ErrorDelete);
  std::string msg = TRITONSERVER_ErrorMessage(err);
  switch (TRITONSERVER_ErrorCode(err)) {
    case TRITONSERVER_ERROR_INTERNAL:
      throw InternalError(std::move(msg));
    case TRITONSERVER_ERROR_NOT_FOUND:
      throw NotFoundError(std::move(msg));
    case TRITONSERVER_ERROR_INVALID_ARG:
      throw InvalidArgumentError(std::move(msg));
    case TRITONSERVER_ERROR_UNAVAILABLE:
      throw UnavailableError(std::move(msg));
    case TRITONSERVER_ERROR_UNSUPPORTED:
      throw UnsupportedError(std::move(msg));
    case TRITONSERVER_ERROR_ALREADY_EXISTS:
      throw AlreadyExistsError(std::move(msg));
    default:
      throw UnknownError(std::move(msg));
  }
}


template <typename Base, typename FrontendServer>
class TritonFrontend {
 private:
  std::shared_ptr<TRITONSERVER_Server> server_;
  std::unique_ptr<Base> service;
  triton::server::RestrictedFeatures restricted_features;
  // TODO: [DLIS-7194] Add support for TraceManager & SharedMemoryManager
  triton::server::TraceManager* trace_manager_;
  // triton::server::SharedMemoryManager shm_manager_;

 public:
  TritonFrontend(uintptr_t server_mem_addr, UnorderedMapType data)
  {
    TRITONSERVER_Server* server_ptr =
        reinterpret_cast<TRITONSERVER_Server*>(server_mem_addr);

    server_.reset(server_ptr, EmptyDeleter);

    std::string trace_file = "/tmp/prashanth.json";
    TRITONSERVER_InferenceTraceLevel trace_level{
        TRITONSERVER_TRACE_LEVEL_TIMESTAMPS};
    int32_t trace_rate{1000};  // Sampling Rate 1/trace_rate requests get traced
    int32_t trace_count{-1};   // Limit of traces collected
    int32_t trace_log_frequency{1};  // Will write to a log for every n traces
    InferenceTraceMode trace_mode{TRACE_MODE_TRITON};  // Or trace_mode_triton
    TraceConfigMap trace_config_map;                   // Can pass trace args

    ThrowIfError(TraceManager::Create(
        &trace_manager_, trace_level, trace_rate, trace_count,
        trace_log_frequency, trace_file, trace_mode, trace_config_map));


    ThrowIfError(FrontendServer::Create(
        server_, data, trace_manager_ /* TraceManager */,
        nullptr /* SharedMemoryManager */, restricted_features, &service));
  };

  // TODO: [DLIS-7194] Add support for TraceManager & SharedMemoryManager
  // TritonFrontend(
  //     uintptr_t server_mem_addr, UnorderedMapType data,
  //     TraceManager trace_manager, SharedMemoryManager shm_manager)

  void StartService() { ThrowIfError(service->Start()); };
  void StopService() { ThrowIfError(service->Stop()); };

  // The frontend does not own the TRITONSERVER_Server* object.
  // Hence, deleting the underlying server instance,
  // will cause a double-free when the core bindings attempt to
  // delete the TRITONSERVER_Server instance.
  static void EmptyDeleter(TRITONSERVER_Server* obj){};

  ~TritonFrontend()
  {
    delete trace_manager_;
    trace_manager_ = nullptr;
  }
};

}}}  // namespace triton::server::python
