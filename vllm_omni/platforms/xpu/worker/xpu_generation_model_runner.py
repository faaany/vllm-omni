# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager

import torch

from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner


@contextmanager
def torch_cuda_wrapper_for_xpu():
    class _EventPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            self.record = lambda: None
            self.synchronize = lambda: None

    try:
        # replace cuda APIs with xpu APIs, this should work by default
        torch.cuda.Event = torch.xpu.Event
        torch.cuda.Stream = torch.xpu.Stream
        torch.cuda.default_stream = torch.xpu.current_stream
        torch.cuda.current_stream = torch.xpu.current_stream
        torch.cuda.stream = torch.xpu.stream
        yield
    finally:
        # if anything goes wrong, just patch it with a placeholder
        torch.cuda.Event = _EventPlaceholder


class XPUGenerationModelRunner(GPUGenerationModelRunner):
    def __init__(self, *args, **kwargs):
        with torch_cuda_wrapper_for_xpu():
            super().__init__(*args, **kwargs)

    def _init_device_properties(self):
        self.num_sms = None

    def _sync_device(self) -> None:
        torch.xpu.synchronize()
