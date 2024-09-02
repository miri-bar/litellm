import json
import time
import types
from typing import Optional, List, Dict, Any, Callable, Union

import httpx  # type: ignore
import requests  # type: ignore

import litellm
from litellm.utils import ModelResponse

from ..base import BaseLLM


class AI21Error(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url="https://api.ai21.com/studio/v1/"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class AI21ChatCompletionConfig:
    """
    Reference: https://docs.ai21.com/reference/jamba-15-api-ref
    """

    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None
    n: Optional[int] = None
    response_format: Optional[Dict[str, str]] = None
    documents: Optional[List[Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        n: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        documents: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    def get_supported_openai_params(self, model: str) -> List:
        return [
            "max_tokens",
            "n",
            "stop",
            "stream",
            "temperature",
            "top_p",
            "tools",
            "tool_choice",
            "function_call",
            "functions",
        ]


def validate_environment(api_key):
    if api_key is None:
        raise ValueError(
            "Missing AI21 API Key - A call is being made to ai21 but no key is set either in the environment variables or via params"
        )
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": "Bearer " + api_key,
    }
    return headers


class AI21ChatCompletion(BaseLLM):
    def __init__(self) -> None:
        super().__init__()

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion: bool = False,
        litellm_params=None,
        logger_fn=None,
        headers={},
        client=None,
    ):
        headers = validate_environment(api_key)

        ## Load Config
        config = litellm.AI21ChatCompletionConfig.get_config()
        for k, v in config.items():
            if (
                k not in optional_params
            ):  # completion(top_k=3) > ai21_config(top_k=3) <- allows for dynamic variables to be passed in
                optional_params[k] = v

        data = {
            "model": model,
            "messages": messages,
            **optional_params,
        }

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "headers": headers,
                "api_base": api_base,
            },
        )

        # TODO: maybe make pop instead of get?
        stream = optional_params.get("stream", False)

        # TODO: tool calling?

        ## ROUTE CALL TO FUNCTION MATCHING THE PARAMETERS
        # if acompletion:
        #     if stream:
        #         # TODO: Async + Stream
        #     else:
        #         # TODO: Async + No Stream
        # else:
        #     if stream:
        #         # TODO: Sync + Stream
        url = (
            api_base
            if api_base.endswith("/chat/completions")
            else api_base + "chat/completions"
        )
        response = requests.post(
            url=url,
            headers=headers,
            data=json.dumps(data),
        )

        ## Error handling for AI21 calls
        if response.status_code != 200:
            raise AI21Error(message=response.text, status_code=response.status_code)

        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=response.text,
            additional_args={"complete_input_dict": data},
        )
        print_verbose(f"raw model_response: {response.text}")

        return self._process_response(
            response=response, model_response=model_response, model=model
        )

    def embedding(self):
        # logic for parsing in - calling - parsing out model embedding calls
        pass

    def _process_response(
        self,
        model: str,
        response: Union[requests.Response, httpx.Response],
        model_response: ModelResponse,
        # stream: bool,
        # logging_obj: litellm.litellm_core_utils.litellm_logging.Logging,
        # optional_params: dict,
        # api_key: str,
        # data: Union[dict, str],
        # messages: List,
        # print_verbose,
        # encoding,
        # json_mode: bool,
    ) -> ModelResponse:
        response_json = response.json()
        model_response.choices = response_json["choices"]
        model_response.created = int(time.time())
        model_response.model = model

        return model_response
