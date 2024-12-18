import asyncio
import random
import time

import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:

    _number_of_response_per_request = 200

    _time_to_first_response = 0.4  # seconds
    _time_to_first_response_std_dev = 0

    _time_between_response = 0.02  # seconds
    _time_between_response_std_dev = 0

    @classmethod
    def auto_complete_config(cls, auto_complete_model_config):
        cls._auto_complete_inputs_and_outputs(auto_complete_model_config)
        auto_complete_model_config.set_model_transaction_policy(dict(decoupled=True))
        auto_complete_model_config.set_max_batch_size(0)
        return auto_complete_model_config

    @staticmethod
    def _auto_complete_inputs_and_outputs(auto_complete_model_config):
        inputs = [
            {"name": "text_input", "data_type": "TYPE_STRING", "dims": [1]},
            {
                "name": "image",
                "data_type": "TYPE_STRING",
                "dims": [-1],  # can be multiple images as separate elements
                "optional": True,
            },
            {
                "name": "stream",
                "data_type": "TYPE_BOOL",
                "dims": [1],
                "optional": True,
            },
            {
                "name": "sampling_parameters",
                "data_type": "TYPE_STRING",
                "dims": [1],
                "optional": True,
            },
            {
                "name": "exclude_input_in_output",
                "data_type": "TYPE_BOOL",
                "dims": [1],
                "optional": True,
            },
            {
                "name": "return_finish_reason",
                "data_type": "TYPE_BOOL",
                "dims": [1],
                "optional": True,
            },
            {
                "name": "return_cumulative_logprob",
                "data_type": "TYPE_BOOL",
                "dims": [1],
                "optional": True,
            },
            {
                "name": "return_logprobs",
                "data_type": "TYPE_BOOL",
                "dims": [1],
                "optional": True,
            },
            {
                "name": "return_num_input_tokens",
                "data_type": "TYPE_BOOL",
                "dims": [1],
                "optional": True,
            },
            {
                "name": "return_num_output_tokens",
                "data_type": "TYPE_BOOL",
                "dims": [1],
                "optional": True,
            },
        ]
        outputs = [
            {"name": "text_output", "data_type": "TYPE_STRING", "dims": [-1]},
            {"name": "finish_reason", "data_type": "TYPE_STRING", "dims": [-1]},
            {"name": "cumulative_logprob", "data_type": "TYPE_FP32", "dims": [-1]},
            {"name": "logprobs", "data_type": "TYPE_STRING", "dims": [-1]},
            {"name": "num_input_tokens", "data_type": "TYPE_UINT32", "dims": [1]},
            {"name": "num_output_tokens", "data_type": "TYPE_UINT32", "dims": [-1]},
        ]

        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []
        for input in config["input"]:
            input_names.append(input["name"])
        for output in config["output"]:
            output_names.append(output["name"])

        for input in inputs:
            if input["name"] not in input_names:
                auto_complete_model_config.add_input(input)
        for output in outputs:
            if output["name"] not in output_names:
                auto_complete_model_config.add_output(output)

    async def execute(self, requests):
        for request in requests:
            start_time = time.time()
            response_sender = request.get_response_sender()
            text_input_tensor = pb_utils.get_input_tensor_by_name(request, "text_input").as_numpy()
            text_input = str(text_input_tensor[0], encoding="utf-8")
            for i in range(self._number_of_response_per_request):
                text_output_tensor = pb_utils.Tensor("text_output", np.array([[" response"]], np.object_))
                response = pb_utils.InferenceResponse(output_tensors=[text_output_tensor])
                await self.delay_between_response(i)
                response_sender.send(response)
            response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
            end_time = time.time()
            pb_utils.Logger.log_info("start-to-finish: " + str(end_time - start_time) + " s")

        return None

    async def delay_between_response(self, i):
        if i == 0:
            delay = random.gauss(mu=self._time_to_first_response, sigma=self._time_to_first_response_std_dev)
        else:
            delay = random.gauss(mu=self._time_between_response, sigma=self._time_between_response_std_dev)
        await asyncio.sleep(delay)
