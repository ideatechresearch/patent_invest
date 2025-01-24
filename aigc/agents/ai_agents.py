import re, json, sys
import inspect, importlib
from threading import Lock
from enum import IntEnum
from copy import deepcopy,copy
from itertools import count, chain, repeat
from typing import Any, Dict, List, Tuple, Callable, Generator, AsyncGenerator, Mapping, Iterable, Optional, Union, \
    get_args, get_origin
from collections import OrderedDict, UserDict, UserList, abc
from pydantic import BaseModel
from dataclasses import dataclass
from functools import wraps, partial
from abc import ABCMeta
from griffe import Docstring, DocstringSectionKind  # 文档字符串解析器


def load_class_from_string(class_path: str, path=None):
    path_in_sys = False
    if path:
        if path not in sys.path:
            path_in_sys = True
            sys.path.insert(0, path)

    try:
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls
    finally:
        if path and path_in_sys:
            sys.path.remove(path)


# 基于配置动态实例化对象，支持类或可调用对象
def create_object(config: Union[Dict, Any] = None):
    """Create an instance based on the configuration where 'type' is a
    preserved key to indicate the class (path). When accepting non-dictionary
    input, the function degenerates to an identity.
    """
    if config is None or not isinstance(config, dict):
        return config
    assert isinstance(config, dict) and 'type' in config

    config = config.copy()
    obj_type = config.pop('type')
    if isinstance(obj_type, str):
        obj_type = load_class_from_string(obj_type)
    if inspect.isclass(obj_type):
        obj = obj_type(**config)
    else:
        assert callable(obj_type)
        obj = partial(obj_type, **config)
    return obj


# need to integrate int, so asdict can convert AgentStatusCode to int
class ModelStatusCode(IntEnum):
    END = 0  # end of streaming
    STREAM_ING = 1  # response is in streaming
    SERVER_ERR = -1  # triton server's error
    SESSION_CLOSED = -2  # session has been closed
    SESSION_OUT_OF_LIMIT = -3  # request length out of limit
    SESSION_INVALID_ARG = -4  # invalid argument
    SESSION_READY = 2  # session is ready for inference


class AgentStatusCode(IntEnum):
    END = 0  # end of streaming
    STREAM_ING = 1  # response is in streaming
    SERVER_ERR = -1  # triton server's error
    SESSION_CLOSED = -2  # session has been closed
    SESSION_OUT_OF_LIMIT = -3  # request length out of limit
    SESSION_INVALID_ARG = -4  # invalid argument
    SESSION_READY = 2  # session is ready for inference
    PLUGIN_START = 3  # start tool
    PLUGIN_END = 4  # finish tool
    PLUGIN_RETURN = 5  # finish tool
    CODING = 6  # start python
    CODE_END = 7  # end python
    CODE_RETURN = 8  # python return


class AgentMessage(BaseModel):
    content: Any
    sender: str = 'user'
    formatted: Optional[Any] = None
    extra_info: Optional[Any] = None
    type: Optional[str] = None
    receiver: Optional[str] = None
    stream_state: Union[ModelStatusCode, AgentStatusCode] = AgentStatusCode.END


class Memory:

    def __init__(self, recent_n: int = None) -> None:
        self.memory: List[AgentMessage] = []
        self.recent_n = recent_n

    def get_memory(
            self,
            recent_n: Optional[int] = None,
            filter_func: Optional[Callable[[int, dict], bool]] = None,
    ) -> list:
        recent_n = recent_n or self.recent_n
        memory = self.memory if recent_n is None else self.memory[-recent_n:]

        if filter_func is not None:
            memory = [m for i, m in enumerate(memory) if filter_func(i, m)]
        return memory

    def add(self, memories: Union[List[Dict], Dict, None]) -> None:
        for memory in memories if isinstance(memories,
                                             (list, tuple)) else [memories]:
            if isinstance(memory, str):
                memory = AgentMessage(sender='user', content=memory)
            if isinstance(memory, AgentMessage):
                self.memory.append(memory)

    def delete(self, index: Union[List, int]) -> None:
        if isinstance(index, int):
            del self.memory[index]
        else:
            for i in index:
                del self.memory[i]

    def load(
            self,
            memories: Union[str, Dict, List],
            overwrite: bool = True,
    ) -> None:
        if overwrite:
            self.memory = []
        if isinstance(memories, dict):
            self.memory.append(AgentMessage(**memories))
        elif isinstance(memories, list):
            for m in memories:
                self.memory.append(AgentMessage(**m))
        else:
            raise TypeError(f'{type(memories)} is not supported')

    def save(self) -> List[dict]:
        memory = []
        for m in self.memory:
            memory.append(m.model_dump())
        return memory


class MemoryManager:

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.memory_map: Dict[int, Memory] = {}

    def create_instance(self, session_id: int):
        self.memory_map[session_id] = create_object(self.cfg)

    def get_memory(self, session_id: int = 0, **kwargs) -> list:
        return self.memory_map[session_id].get_memory(**kwargs)

    def add(self, memory, session_id: int = 0, **kwargs) -> None:
        if session_id not in self.memory_map:
            self.create_instance(session_id)
        self.memory_map[session_id].add(memory, **kwargs)

    def get(self, session_id: int = 0) -> Memory:
        return self.memory_map.get(session_id, None)

    def reset(self, session_id: int = 0) -> None:
        if session_id in self.memory_map:
            del self.memory_map[session_id]


class DefaultAggregator:

    def aggregate(self,
                  messages: Memory,
                  name: str,
                  system_instruction: str = None) -> List[Dict[str, str]]:
        _message = []
        messages = messages.get_memory()
        if system_instruction:
            _message.extend(
                self.aggregate_system_intruction(system_instruction))
        for message in messages:
            if message.sender == name:
                _message.append(dict(role='assistant', content=str(message.content)))
            else:
                user_message = message.content
                if len(_message) > 0 and _message[-1]['role'] == 'user':
                    _message[-1]['content'] += user_message
                else:
                    _message.append(dict(role='user', content=user_message))
        return _message

    @staticmethod
    def aggregate_system_intruction(system_intruction) -> List[dict]:
        if isinstance(system_intruction, str):
            system_intruction = dict(role='system', content=system_intruction)
        if isinstance(system_intruction, dict):
            system_intruction = [system_intruction]
        if isinstance(system_intruction, list):
            for msg in system_intruction:
                if not isinstance(msg, dict):
                    raise TypeError(f'Unsupported message type: {type(msg)}')
                if not ('role' in msg and 'content' in msg):
                    raise KeyError(f"Missing required key 'role' or 'content': {msg}")
        return system_intruction


class RemovableHandle:
    _id_iter = count(0)

    def __init__(self, hooks_dict):
        self.hooks_dict = hooks_dict
        self.id = next(self._id_iter)

    def remove(self):
        """
        Removes the hook associated with this handle from the hooks dictionary.
        """
        if self.id in self.hooks_dict:
            del self.hooks_dict[self.id]


class Hook:
    def before(self, agent, message: Tuple['AgentMessage'], session_id: int):
        """
        This method is executed before the agent processes the message.
        """
        # 可以在这里处理 before 操作
        pass

    def after(self, agent, message: 'AgentMessage', session_id: int):
        """
        This method is executed after the agent processes the message.
        """
        # 可以在这里处理 after 操作
        pass


class Agent:

    def __init__(self, memory: Dict = None,
                 hooks: Optional[Union[List[Dict], Dict]] = None,
                 name: Optional[str] = None,
                 ):

        self.name = name or self.__class__.__name__
        self.memory: MemoryManager = MemoryManager(memory) if memory else None
        self._hooks: Dict[int, Hook] = OrderedDict()

        if hooks:
            for hook in hooks:
                hook = create_object(hook)
                self.register_hook(hook)

    def register_hook(self, hook: Callable):
        """
        Register a new hook and return a RemovableHandle that can be used to remove the hook.
        """
        handle = RemovableHandle(self._hooks)
        self._hooks[handle.id] = hook
        return handle

    def before(self, *message: AgentMessage, session_id: int = 0):
        """
        Trigger all registered hooks.
        """
        message = [AgentMessage(sender='user', content=m) if isinstance(m, str) else deepcopy(m) for m in message]
        for hook in self._hooks.values():
            result = hook.before(self, message, session_id)
            if result:
                message = result

        if self.memory:
            self.memory.add(message, session_id=session_id)

    def after(self, response_message, session_id: int = 0):
        """
        Trigger all registered hooks.
        """
        if self.memory:
            self.memory.add(response_message, session_id=session_id)

        response_message = deepcopy(response_message)
        for hook in self._hooks.values():
            result = hook.after(self, response_message, session_id)
            if result:
                response_message = result

        return response_message

    def forward(self, *message: AgentMessage, llm: Callable, **kwargs) -> Union[AgentMessage, str]:
        # formatted_messages
        response_message = llm(message, **kwargs)
        # formatted_messages
        if not isinstance(response_message, AgentMessage):
            response_message = AgentMessage(sender=self.name, content=response_message)

        return response_message

    def __call__(self, *message: AgentMessage, session_id=0, **kwargs) -> AgentMessage:
        self.before(*message, session_id=session_id)
        response_message = self.forward(*message, **kwargs)
        return self.after(response_message, session_id=session_id)

    # 管理子代理的注册，建立代理间的层次化关系
    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, Agent):
            _agents = getattr(self, '_agents', OrderedDict())
            _agents[__name] = __value
            super().__setattr__('_agents', _agents)
        super().__setattr__(__name, __value)

    # 显示代理及其子代理的层次结构
    def __repr__(self):

        def _rcsv_repr(agent, n_indent=1):
            res = agent.__class__.__name__ + (f"(name='{agent.name}')" if agent.name else '')
            modules = [
                f"{n_indent * '  '}({name}): {_rcsv_repr(agent, n_indent + 1)}"
                for name, agent in getattr(agent, '_agents', {}).items()
            ]
            if modules:
                res += '(\n' + '\n'.join(modules) + f'\n{(n_indent - 1) * "  "})'
            elif not res.endswith(')'):
                res += '()'
            return res

        return _rcsv_repr(self)


class AsyncStreamingAgent(Agent):
    """Component that makes asynchronous agent calling output a streaming response."""

    async def forward(self, *message: AgentMessage, llm: Callable, **kwargs) -> AsyncGenerator[
        Union[AgentMessage, Tuple[ModelStatusCode, str]], None]:
        async for model_state, response, *_ in llm(message, **kwargs):
            response_message = AgentMessage(sender=self.name, content=response, stream_state=model_state)
            yield response_message

    async def __call__(self, *message: AgentMessage, session_id=0, **kwargs) -> AsyncGenerator[AgentMessage, None]:
        self.before(*message, session_id=session_id)
        response_message = AgentMessage(sender=self.name, content="")
        async for response_message in self.forward(*message, **kwargs):
            yield response_message.model_copy()

        response_message = self.after(response_message, session_id=session_id)
        yield response_message


class AgentContainer:

    def __init_subclass__(cls):
        super().__init_subclass__()

        def wrap_api(func):

            @wraps(func)
            def wrapped_func(self, *args, **kwargs):
                data = self.data.copy() if hasattr(self, 'data') else None  # 备份 self.data

                def _backup(d):
                    if d is None:
                        self.data.clear()
                    else:
                        self.data = d

                ret = func(self, *args, **kwargs)
                agents = OrderedDict()
                for k, item in self.data.items() if isinstance(self.data, abc.Mapping) else enumerate(self.data):
                    if isinstance(self.data, abc.Mapping) and not isinstance(k, str):
                        _backup(data)
                        raise KeyError(f'agent name should be a string, got {type(k)}')
                    if isinstance(k, str) and '.' in k:  # 键必须是字符串且不包含 .
                        _backup(data)
                        raise KeyError(f'agent name can\'t contain ".", got {k}')
                    if not isinstance(item, Agent):  # 值必须是 Agent 的实例或子类
                        _backup(data)
                        raise TypeError(f'{type(item)} is not an Agent subclass')
                    agents[str(k)] = item
                self._agents = agents
                return ret

            return wrapped_func

        # fmt: off
        for method in [
            'append', 'sort', 'reverse', 'pop', 'clear', 'update',
            'insert', 'extend', 'remove', '__init__', '__setitem__',
            '__delitem__', '__add__', '__iadd__', '__radd__', '__mul__',
            '__imul__', '__rmul__'
        ]:
            if hasattr(cls, method):
                setattr(cls, method, wrap_api(getattr(cls, method)))


class AgentList(Agent, UserList, AgentContainer):

    def __init__(self, agents: Optional[Iterable[Agent]] = None):
        Agent.__init__(self, memory=None)
        UserList.__init__(self, agents)
        self.name = None


class Sequential(Agent):
    """Sequential is an agent container that forwards messages to each agent
    in the order they are added.
    接收消息，依次传递给内部子代理，直到达到指定的代理或处理完所有代理,允许动态添加和访问子代理。
    """

    def __init__(self, *agents: Union[Agent, Iterable], **kwargs):
        super().__init__(**kwargs)
        self._agents = OrderedDict()
        if not agents:
            raise ValueError('At least one agent should be provided')
        if isinstance(agents[0], Iterable) and not isinstance(agents[0], Agent):
            if not agents[0]:
                raise ValueError('At least one agent should be provided')
            agents = agents[0]
        for key, agent in enumerate(agents):
            if isinstance(agents, Mapping):
                key, agent = agent, agents[agent]
            elif isinstance(agent, tuple):
                key, agent = agent
            self.add_agent(key, agent)

    def add_agent(self, name: str, agent: Agent):
        assert isinstance(agent, Agent), f'{type(agent)} is not an Agent subclass'
        self._agents[str(name)] = agent

    # 消息的顺序流转
    def forward(self, *message: AgentMessage, session_id=0, exit_at: Optional[int] = None, **kwargs) -> AgentMessage:
        assert exit_at is None or exit_at >= 0, 'exit_at should be greater than or equal to 0'
        if exit_at is None:
            exit_at = len(self) - 1

        iterator = chain.from_iterable(repeat(self._agents.values()))
        for _ in range(exit_at + 1):
            agent = next(iterator)
            if isinstance(message, AgentMessage):
                message = (message,)
            message = agent(*message, session_id=session_id, **kwargs)
        return message

    def __getitem__(self, key):
        if isinstance(key, int) and key < 0:
            assert key >= -len(self), 'index out of range'
            key = len(self) + key
        return self._agents[str(key)]

    def __len__(self):
        return len(self._agents)


class AsyncStreamingSequential(AsyncStreamingAgent, Sequential):
    """
    Streaming variant of the AsyncSequential class
    代理以 流式方式 处理消息并异步传递结果
    """

    async def forward(self, *message: AgentMessage, session_id=0, exit_at: Optional[int] = None, **kwargs):
        assert exit_at is None or exit_at >= 0, 'exit_at should be greater than or equal to 0'
        if exit_at is None:
            exit_at = len(self) - 1
        iterator = chain.from_iterable(repeat(self._agents.values()))
        for _ in range(exit_at + 1):
            agent = next(iterator)
            if isinstance(message, AgentMessage):
                message = (message,)
            # 逐步从每个子代理中获取处理结果
            async for message in agent(*message, session_id=session_id, **kwargs):
                yield message


# agent_list = AgentList([Agent(name="Agent1"), Agent(name="Agent2")])
# async_seq = AsyncStreamingSequential(agent_list )
# async def process():
#     async for response in async_seq.forward(AgentMessage(data="Hello")):
#         print(response)
#
#
# class BaseLLM:
#     """Base class for model wrapper.
#
#     Args:
#         path (str): The path to the model.
#         max_new_tokens (int): Maximum length of output expected to be generated by the model. Defaults
#             to 512.
#         tokenizer_only (bool): If True, only the tokenizer will be initialized.
#             Defaults to False.
#         meta_template (list of dict, optional): The model's meta prompt
#             template if needed, in case the requirement of injecting or
#             wrapping of any meta instructions.
#     """
#
#     def __init__(self,
#                  path: str,
#                  model_type: str,
#                  retry: int = 2,
#                  tokenizer_only: bool = False,
#                  meta_template: Optional[List[Dict]] = None,
#                  *,
#                  max_new_tokens: int = 512,
#                  top_p: float = 0.8,
#                  top_k: float = 40,
#                  temperature: float = 0.8,
#                  repetition_penalty: float = 1.0,
#                  stop_words: Union[List[str], str] = None):
#         self.path = path
#         self.tokenizer_only = tokenizer_only
#         self.model_type = model_type
#         self.retry = retry
#         self.eos_token_id = None
#         if meta_template and 'eos_token_id' in meta_template:
#             self.eos_token_id = meta_template['eos_token_id']
#
#         if isinstance(stop_words, str):
#             stop_words = [stop_words]
#         self.gen_params = dict(
#             max_new_tokens=max_new_tokens,
#             top_p=top_p,
#             top_k=top_k,
#             temperature=temperature,
#             repetition_penalty=repetition_penalty,
#             stop_words=stop_words)
#
#     def generate(self, inputs: Union[str, List[str]], **gen_params) -> str:
#         raise NotImplementedError
#
#     def stream_generate(self, inputs: str, **gen_params) -> List[str]:
#         raise NotImplementedError
#
#     def chat(self,
#              inputs: Union[List[dict], List[List[dict]]],
#              session_ids: Union[int, List[int]] = None,
#              **gen_params):
#         if isinstance(inputs[0], list):
#             _inputs = list()
#             for msg in inputs:
#                 _inputs.append(self.template_parser(msg))
#         else:
#             _inputs = self.template_parser(inputs)
#         return self.generate(_inputs, **gen_params)
#
#     def stream_chat(self, inputs: List[dict], **gen_params):
#         raise NotImplementedError
#
#     def tokenize(self, prompts: Union[str, List[str], List[dict],
#                                       List[List[dict]]]):
#
#         raise NotImplementedError
#
#     def update_gen_params(self, **kwargs):
#         gen_params = copy(self.gen_params)
#         gen_params.update(kwargs)
#         return gen_params
#
#
#
# class AsyncLLM:
#
#     async def generate(self,
#                        inputs: Union[str, List[str]],
#                        session_ids: Union[int, List[int]] = None,
#                        **gen_params) -> str:
#
#         raise NotImplementedError
#
#     async def stream_generate(self, inputs: str, **gen_params) -> List[str]:
#
#         raise NotImplementedError
#
#     async def chat(self,
#                    inputs: Union[List[dict], List[List[dict]]],
#                    session_ids: Union[int, List[int]] = None,
#                    **gen_params):
#
#         if isinstance(inputs[0], list):
#             _inputs = list()
#             for msg in inputs:
#                 _inputs.append(self.template_parser(msg))
#         else:
#             _inputs = self.template_parser(inputs)
#         return await self.generate(_inputs, session_ids, **gen_params)
#
#     async def stream_chat(self, inputs: List[dict], **gen_params):
#         raise NotImplementedError
#
#     async def tokenize(self, prompts: Union[str, List[str], List[dict],List[List[dict]]]):
#
#         raise NotImplementedError
#
# class AsyncBaseAPILLM(AsyncLLM, BaseLLM):
#     pass

# class AsyncGPTAPI(AsyncBaseAPILLM):
#     """Model wrapper around OpenAI's models.
#
#     Args:
#         model_type (str): The name of OpenAI's model.
#         retry (int): Number of retires if the API call fails. Defaults to 2.
#         key (str or List[str]): OpenAI key(s). In particular, when it
#             is set to "ENV", the key will be fetched from the environment
#             variable $OPENAI_API_KEY, as how openai defaults to be. If it's a
#             list, the keys will be used in round-robin manner. Defaults to
#             'ENV'.
#         org (str or List[str], optional): OpenAI organization(s). If not
#             specified, OpenAI uses the default organization bound to each API
#             key. If specified, the orgs will be posted with each request in
#             round-robin manner. Defaults to None.
#         meta_template (Dict, optional): The model's meta prompt
#             template if needed, in case the requirement of injecting or
#             wrapping of any meta instructions.
#         api_base (str): The base url of OpenAI's API. Defaults to
#             'https://api.openai.com/v1/chat/completions'.
#         gen_params: Default generation configuration which could be overridden
#             on the fly of generation.
#     """
#
#     is_api: bool = True
#
#     def __init__(self,
#                  model_type: str = 'gpt-3.5-turbo',
#                  retry: int = 2,
#                  json_mode: bool = False,
#                  key: Union[str, List[str]] = 'ENV',
#                  org: Optional[Union[str, List[str]]] = None,
#                  meta_template: Optional[Dict] = [
#                      dict(role='system', api_role='system'),
#                      dict(role='user', api_role='user'),
#                      dict(role='assistant', api_role='assistant')
#                  ],
#                  api_base: str = OPENAI_API_BASE,
#                  proxies: Optional[Dict] = None,
#                  **gen_params):
#         if 'top_k' in gen_params:
#             warnings.warn('`top_k` parameter is deprecated in OpenAI APIs.',
#                           DeprecationWarning)
#             gen_params.pop('top_k')
#         super().__init__(
#             model_type=model_type,
#             meta_template=meta_template,
#             retry=retry,
#             **gen_params)
#         self.gen_params.pop('top_k')
#         self.logger = getLogger(__name__)
#
#         if isinstance(key, str):
#             self.keys = [os.getenv('OPENAI_API_KEY') if key == 'ENV' else key]
#         else:
#             self.keys = key
#
#         # record invalid keys and skip them when requesting API
#         # - keys have insufficient_quota
#         self.invalid_keys = set()
#
#         self.key_ctr = 0
#         if isinstance(org, str):
#             self.orgs = [org]
#         else:
#             self.orgs = org
#         self.org_ctr = 0
#         self.url = api_base
#         self.model_type = model_type
#         self.proxies = proxies or {}
#         self.json_mode = json_mode
#
#     async def chat(
#         self,
#         inputs: Union[List[dict], List[List[dict]]],
#         session_ids: Union[int, List[int]] = None,
#         **gen_params,
#     ) -> Union[str, List[str]]:
#         """Generate responses given the contexts.
#
#         Args:
#             inputs (Union[List[dict], List[List[dict]]]): a list of messages
#                 or list of lists of messages
#             gen_params: additional generation configuration
#
#         Returns:
#             Union[str, List[str]]: generated string(s)
#         """
#         assert isinstance(inputs, list)
#         if 'max_tokens' in gen_params:
#             raise NotImplementedError('unsupported parameter: max_tokens')
#         gen_params = {**self.gen_params, **gen_params}
#         tasks = [
#             self._chat(messages, **gen_params) for messages in (
#                 [inputs] if isinstance(inputs[0], dict) else inputs)
#         ]
#         ret = await asyncio.gather(*tasks)
#         return ret[0] if isinstance(inputs[0], dict) else ret
#
#     async def stream_chat(
#         self,
#         inputs: List[dict],
#         **gen_params,
#     ):
#         """Generate responses given the contexts.
#
#         Args:
#             inputs (List[dict]): a list of messages
#             gen_params: additional generation configuration
#
#         Returns:
#             str: generated string
#         """
#         assert isinstance(inputs, list)
#         if 'max_tokens' in gen_params:
#             raise NotImplementedError('unsupported parameter: max_tokens')
#         gen_params = self.update_gen_params(**gen_params)
#         gen_params['stream'] = True
#
#         resp = ''
#         finished = False
#         stop_words = gen_params.get('stop_words')
#         if stop_words is None:
#             stop_words = []
#         # mapping to role that openai supports
#         messages = self.template_parser._prompt2api(inputs)
#         async for text in self._stream_chat(messages, **gen_params):
#             if self.model_type.lower().startswith('qwen'):
#                 resp = text
#             else:
#                 resp += text
#             if not resp:
#                 continue
#             # remove stop_words
#             for sw in stop_words:
#                 if sw in resp:
#                     resp = filter_suffix(resp, stop_words)
#                     finished = True
#                     break
#             yield ModelStatusCode.STREAM_ING, resp, None
#             if finished:
#                 break
#         yield ModelStatusCode.END, resp, None
#
#     async def _chat(self, messages: List[dict], **gen_params) -> str:
#         """Generate completion from a list of templates.
#
#         Args:
#             messages (List[dict]): a list of prompt dictionaries
#             gen_params: additional generation configuration
#
#         Returns:
#             str: The generated string.
#         """
#         assert isinstance(messages, list)
#
#         header, data = self.generate_request_data(
#             model_type=self.model_type,
#             messages=messages,
#             gen_params=gen_params,
#             json_mode=self.json_mode)
#
#         max_num_retries, errmsg = 0, ''
#         while max_num_retries < self.retry:
#             if len(self.invalid_keys) == len(self.keys):
#                 raise RuntimeError('All keys have insufficient quota.')
#
#             # find the next valid key
#             while True:
#                 self.key_ctr += 1
#                 if self.key_ctr == len(self.keys):
#                     self.key_ctr = 0
#
#                 if self.keys[self.key_ctr] not in self.invalid_keys:
#                     break
#
#             key = self.keys[self.key_ctr]
#             header['Authorization'] = f'Bearer {key}'
#
#             if self.orgs:
#                 self.org_ctr += 1
#                 if self.org_ctr == len(self.orgs):
#                     self.org_ctr = 0
#                 header['OpenAI-Organization'] = self.orgs[self.org_ctr]
#
#             response = dict()
#             try:
#                 async with aiohttp.ClientSession() as session:
#                     async with session.post(
#                             self.url,
#                             headers=header,
#                             json=data,
#                             proxy=self.proxies.get(
#                                 'https', self.proxies.get('http'))) as resp:
#                         response = await resp.json()
#                         return response['choices'][0]['message'][
#                             'content'].strip()
#             except aiohttp.ClientConnectionError:
#                 errmsg = 'Got connection error ' + str(traceback.format_exc())
#                 self.logger.error(errmsg)
#                 continue
#             except aiohttp.ClientResponseError as e:
#                 errmsg = 'Response error, got ' + str(e)
#                 self.logger.error(errmsg)
#                 continue
#             except json.JSONDecodeError:
#                 errmsg = 'JsonDecode error, got ' + (await resp.text(
#                     errors='replace'))
#                 self.logger.error(errmsg)
#                 continue
#             except KeyError:
#                 if 'error' in response:
#                     if response['error']['code'] == 'rate_limit_exceeded':
#                         time.sleep(1)
#                         continue
#                     elif response['error']['code'] == 'insufficient_quota':
#                         self.invalid_keys.add(key)
#                         self.logger.warn(f'insufficient_quota key: {key}')
#                         continue
#
#                     errmsg = 'Find error message in response: ' + str(
#                         response['error'])
#                     self.logger.error(errmsg)
#             except Exception as error:
#                 errmsg = str(error) + '\n' + str(traceback.format_exc())
#                 self.logger.error(errmsg)
#             max_num_retries += 1
#
#         raise RuntimeError('Calling OpenAI failed after retrying for '
#                            f'{max_num_retries} times. Check the logs for '
#                            f'details. errmsg: {errmsg}')
#
#     async def _stream_chat(self, messages: List[dict],
#                            **gen_params) -> AsyncGenerator[str, None]:
#         """Generate completion from a list of templates.
#
#         Args:
#             messages (List[dict]): a list of prompt dictionaries
#             gen_params: additional generation configuration
#
#         Returns:
#             str: The generated string.
#         """
#
#         async def streaming(raw_response):
#             async for chunk in raw_response.content:
#                 if chunk:
#                     decoded = chunk.decode('utf-8')
#                     if decoded.startswith('data: [DONE]'):
#                         return
#                     if decoded[:5] == 'data:':
#                         decoded = decoded[5:]
#                         if decoded[0] == ' ':
#                             decoded = decoded[1:]
#                     else:
#                         print(decoded)
#                         continue
#                     try:
#                         response = json.loads(decoded)
#                         if 'code' in response and response['code'] == -20003:
#                             # Context exceeds maximum length
#                             yield ''
#                             return
#                         if self.model_type.lower().startswith('qwen'):
#                             choice = response['output']['choices'][0]
#                             yield choice['message']['content']
#                             if choice['finish_reason'] == 'stop':
#                                 return
#                         else:
#                             choice = response['choices'][0]
#                             if choice['finish_reason'] == 'stop':
#                                 return
#                             yield choice['delta'].get('content', '')
#                     except Exception as exc:
#                         msg = f'response {decoded} lead to exception of {str(exc)}'
#                         self.logger.error(msg)
#                         raise Exception(msg) from exc
#
#         assert isinstance(messages, list)
#
#         header, data = self.generate_request_data(
#             model_type=self.model_type,
#             messages=messages,
#             gen_params=gen_params,
#             json_mode=self.json_mode)
#
#         max_num_retries, errmsg = 0, ''
#         while max_num_retries < self.retry:
#             if len(self.invalid_keys) == len(self.keys):
#                 raise RuntimeError('All keys have insufficient quota.')
#
#             # find the next valid key
#             while True:
#                 self.key_ctr += 1
#                 if self.key_ctr == len(self.keys):
#                     self.key_ctr = 0
#
#                 if self.keys[self.key_ctr] not in self.invalid_keys:
#                     break
#
#             key = self.keys[self.key_ctr]
#             header['Authorization'] = f'Bearer {key}'
#
#             if self.orgs:
#                 self.org_ctr += 1
#                 if self.org_ctr == len(self.orgs):
#                     self.org_ctr = 0
#                 header['OpenAI-Organization'] = self.orgs[self.org_ctr]
#
#             response = dict()
#             try:
#                 async with aiohttp.ClientSession() as session:
#                     async with session.post(
#                             self.url,
#                             headers=header,
#                             json=data,
#                             proxy=self.proxies.get(
#                                 'https',
#                                 self.proxies.get('http'))) as raw_response:
#                         async for msg in streaming(raw_response):
#                             yield msg
#                         return
#             except aiohttp.ClientConnectionError:
#                 errmsg = 'Got connection error ' + str(traceback.format_exc())
#                 self.logger.error(errmsg)
#                 continue
#             except aiohttp.ClientResponseError as e:
#                 errmsg = 'Response error, got ' + str(e)
#                 self.logger.error(errmsg)
#                 continue
#             except KeyError:
#                 if 'error' in response:
#                     if response['error']['code'] == 'rate_limit_exceeded':
#                         time.sleep(1)
#                         continue
#                     elif response['error']['code'] == 'insufficient_quota':
#                         self.invalid_keys.add(key)
#                         self.logger.warn(f'insufficient_quota key: {key}')
#                         continue
#
#                     errmsg = 'Find error message in response: ' + str(
#                         response['error'])
#                     self.logger.error(errmsg)
#             except Exception as error:
#                 errmsg = str(error) + '\n' + str(traceback.format_exc())
#                 self.logger.error(errmsg)
#             max_num_retries += 1
#
#         raise RuntimeError('Calling OpenAI failed after retrying for '
#                            f'{max_num_retries} times. Check the logs for '
#                            f'details. errmsg: {errmsg}')
#
#     def generate_request_data(self,
#                               model_type,
#                               messages,
#                               gen_params,
#                               json_mode=False):
#         """
#         Generates the request data for different model types.
#
#         Args:
#             model_type (str): The type of the model (e.g., 'gpt', 'internlm', 'qwen').
#             messages (list): The list of messages to be sent to the model.
#             gen_params (dict): The generation parameters.
#             json_mode (bool): Flag to determine if the response format should be JSON.
#
#         Returns:
#             tuple: A tuple containing the header and the request data.
#         """
#         # Copy generation parameters to avoid modifying the original dictionary
#         gen_params = gen_params.copy()
#
#         # Hold out 100 tokens due to potential errors in token calculation
#         max_tokens = min(gen_params.pop('max_new_tokens'), 4096)
#         if max_tokens <= 0:
#             return '', ''
#
#         # Initialize the header
#         header = {
#             'content-type': 'application/json',
#         }
#
#         # Common parameters processing
#         gen_params['max_tokens'] = max_tokens
#         if 'stop_words' in gen_params:
#             gen_params['stop'] = gen_params.pop('stop_words')
#         if 'repetition_penalty' in gen_params:
#             gen_params['frequency_penalty'] = gen_params.pop(
#                 'repetition_penalty')
#
#         # Model-specific processing
#         data = {}
#         if model_type.lower().startswith('gpt'):
#             if 'top_k' in gen_params:
#                 warnings.warn(
#                     '`top_k` parameter is deprecated in OpenAI APIs.',
#                     DeprecationWarning)
#                 gen_params.pop('top_k')
#             gen_params.pop('skip_special_tokens', None)
#             gen_params.pop('session_id', None)
#             data = {
#                 'model': model_type,
#                 'messages': messages,
#                 'n': 1,
#                 **gen_params
#             }
#             if json_mode:
#                 data['response_format'] = {'type': 'json_object'}
#         elif model_type.lower().startswith('internlm'):
#             data = {
#                 'model': model_type,
#                 'messages': messages,
#                 'n': 1,
#                 **gen_params
#             }
#             if json_mode:
#                 data['response_format'] = {'type': 'json_object'}
#         elif model_type.lower().startswith('qwen'):
#             header['X-DashScope-SSE'] = 'enable'
#             gen_params.pop('skip_special_tokens', None)
#             gen_params.pop('session_id', None)
#             if 'frequency_penalty' in gen_params:
#                 gen_params['repetition_penalty'] = gen_params.pop(
#                     'frequency_penalty')
#             gen_params['result_format'] = 'message'
#             data = {
#                 'model': model_type,
#                 'input': {
#                     'messages': messages
#                 },
#                 'parameters': {
#                     **gen_params
#                 }
#             }
#         else:
#             raise NotImplementedError(
#                 f'Model type {model_type} is not supported')
#
#         return header, data
#
#     def tokenize(self, prompt: str) -> list:
#         """Tokenize the input prompt.
#
#         Args:
#             prompt (str): Input string.
#
#         Returns:
#             list: token ids
#         """
#         import tiktoken
#         self.tiktoken = tiktoken
#         enc = self.tiktoken.encoding_for_model(self.model_type)
#         return enc.encode(prompt)


class ActionStatusCode(IntEnum):
    ING = 1
    SUCCESS = 0
    HTTP_ERROR = -1000  # http error
    ARGS_ERROR = -1001  # parameter error
    API_ERROR = -1002  # unknown error


class ActionValidCode(IntEnum):
    FINISH = 1
    OPEN = 0
    CLOSED = -1
    INVALID = -2
    ABSENT = -3  # NO ACTION


@dataclass
class ActionReturn:
    args: Optional[dict] = None
    url: Optional[str] = None
    type: Optional[str] = None
    result: Optional[List[dict]] = None
    errmsg: Optional[str] = None
    state: Union[ActionStatusCode, int] = ActionStatusCode.SUCCESS
    thought: Optional[str] = None
    valid: Optional[ActionValidCode] = ActionValidCode.OPEN

    def format_result(self) -> str:
        """Concatenate items in result."""
        result = []
        for item in self.result or []:
            if item['type'] == 'text':
                result.append(item['content'])
            else:
                result.append(f"[{item['type']}]({item['content']})")
        result = '\n'.join(result)
        return result


def tool_api(func: Optional[Callable] = None,
             *,
             explode_return: bool = False,
             returns_named_value: bool = False,
             **kwargs):
    """将函数转化为API工具，解析其类型注解和文档字符串。

        Args:
            func (Optional[Callable]): 需要装饰的函数，默认 None。
            explode_return (bool): 是否展开返回值为字典的字段，默认为 False。
            returns_named_value (bool): 是否解析返回值中的“名称: 描述”结构，默认为 False。
            **kwargs: 其他参数传递到文档解析器。
    """
    type_map = {"list": "Array", "float": "FLOAT", "int": "NUMBER", "bool": "BOOLEAN", "str": "STRING"}

    def _detect_type(annotation: str) -> str:
        """检测类型，根据字符串返回类型描述"""
        return next((v for k, v in type_map.items() if k in annotation), "STRING")

    def parse_docstring(doc: str):
        """解析函数文档字符串，返回描述、参数和返回值信息"""
        docs = Docstring(doc).parse("google", returns_named_value=returns_named_value, **kwargs)
        result = {"description": "", "parameters": [], "returns": []}
        returns_doc = []
        for section in docs:
            if section.kind is DocstringSectionKind.text:
                result["description"] = section.value
            elif section.kind is DocstringSectionKind.parameters:
                result["parameters"] = [
                    {
                        "name": param.name,
                        "type": _detect_type(param.annotation.lower()) if param.annotation else "STRING",
                        "description": param.description or "",
                    }
                    for param in section.value]
            elif section.kind is DocstringSectionKind.returns:
                result["returns"] = [
                    {
                        "type": _detect_type(ret.annotation.lower()) if ret.annotation else "STRING",
                        "description": ret.description or "",
                    }
                    for ret in section.value]
                returns_doc += [ret for ret in section.value]
        return result, returns_doc

    def _explode(desc):
        """解析描述并提取参数信息为结构化数据."""
        kvs = []
        desc = '\nArgs:\n' + '\n'.join([
            '    ' + item.lstrip(' -+*#.')
            for item in desc.split('\n')[1:] if item.strip()
        ])
        docs = Docstring(desc).parse('google')
        if not docs or docs[0].kind is not DocstringSectionKind.parameters:
            return kvs

        for d in docs[0].value:
            d = d.as_dict()
            if not d['annotation']:
                d.pop('annotation', None)
            else:
                d['type'] = _detect_type(d.pop('annotation').lower())
            kvs.append(d)
        return kvs

    def parse_function(func):
        """解析函数签名及文档，生成描述"""
        doc = func.__doc__ or ""
        doc_info, returns_doc = parse_docstring(re.sub(r":(.+?):`(.+?)`", r"\2", doc))
        sig = inspect.signature(func)
        # print(doc_info, returns_doc )

        parameters = []
        required = []

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            param_type = doc_info.get(param.name, {}).get('type', 'STRING')
            if param.annotation is not inspect.Signature.empty:
                annotation = param.annotation
                while get_origin(annotation):
                    annotation = get_args(annotation)
                param_type = _detect_type(str(annotation))

            parameters.append({"name": name,  # param.name
                               "type": param_type,
                               "description": doc_info.get(param.name, {}).get("description", "")})

            if param.default is inspect.Signature.empty:
                required.append(name)

        desc = {
            "name": func.__name__,
            "description": doc_info["description"],
            "parameters": parameters,
            "required": required,
            "returns": doc_info["returns"],
        }
        return_data = _explode(returns_doc[0]['description']) if explode_return else returns_doc
        if return_data:
            desc['return_data'] = return_data
        return desc

    def api_decorator(target_func):
        """装饰器，动态生成函数描述并附加到函数"""
        func.api_description = parse_function(target_func)

        @wraps(target_func)
        async def async_wrapper(*args, **kwargs):
            return await target_func(*args, **kwargs)

        @wraps(target_func)
        def sync_wrapper(*args, **kwargs):
            return target_func(*args, **kwargs)

        return async_wrapper if inspect.iscoroutinefunction(target_func) else sync_wrapper

    if callable(func):
        return api_decorator(func)

    return api_decorator


class ToolMeta(ABCMeta):
    """
    Metaclass of tools.
    为所有工具类提供统一的元数据管理和API注册逻辑。

    核心功能：
        解析类中的所有方法，自动识别被 tool_api 装饰的函数。
        提取并存储工具类的描述信息（如名称、参数、返回值、支持的API列表等）。
        检查类的接口定义是否符合规范（例如：run 方法与其他API不能同时存在）。

    适用场景：
        构建工具类或工具包（包含多个API）的框架。
        动态生成工具类的接口描述，用于注册到系统或文档中。
    """

    def __new__(mcs, name, base, attrs):
        is_toolkit, tool_desc = True, dict(
            name=name,
            description=Docstring(attrs.get('__doc__',
                                            '')).parse('google')[0].value)
        for key, value in attrs.items():
            if callable(value) and hasattr(value, 'api_description'):
                api_desc = getattr(value, 'api_description')
                if key == 'run':
                    tool_desc['parameters'] = api_desc['parameters']
                    tool_desc['required'] = api_desc['required']
                    if api_desc['description']:
                        tool_desc['description'] = api_desc['description']
                    if api_desc.get('return_data'):
                        tool_desc['return_data'] = api_desc['return_data']
                    is_toolkit = False
                else:
                    tool_desc.setdefault('api_list', []).append(api_desc)

        if not is_toolkit and 'api_list' in tool_desc:
            raise KeyError('`run` and other tool APIs can not be implemented '
                           'at the same time')
        if is_toolkit and 'api_list' not in tool_desc:
            is_toolkit = False
            if callable(attrs.get('run')):
                run_api = tool_api(attrs['run'])
                api_desc = run_api.api_description
                tool_desc['parameters'] = api_desc['parameters']
                tool_desc['required'] = api_desc['required']
                if api_desc['description']:
                    tool_desc['description'] = api_desc['description']
                if api_desc.get('return_data'):
                    tool_desc['return_data'] = api_desc['return_data']
                attrs['run'] = run_api
            else:
                tool_desc['parameters'], tool_desc['required'] = [], []

        attrs['_is_toolkit'] = is_toolkit
        attrs['__tool_description__'] = tool_desc
        return super().__new__(mcs, name, base, attrs)


class JsonParser:
    """Json parser to convert input string into a dictionary.

    Args:
        action (:class:`BaseAction`): action to validate
    """

    PARAMETER_DESCRIPTION = (
        'If you call this tool, you must pass arguments in '
        'the JSON format {key: value}, where the key is the parameter name.')

    def __init__(self, action):
        self.action = action
        self._api2param = {}
        self._api2required = {}
        # perform basic argument validation
        if action.description:
            for api in action.description.get('api_list', [action.description]):
                name = (f'{action.name}.{api["name"]}' if self.action.is_toolkit else api['name'])
                required_parameters = set(api['required'])
                all_parameters = {j['name'] for j in api['parameters']}
                if not required_parameters.issubset(all_parameters):
                    raise ValueError(
                        f'unknown parameters for function "{name}": {required_parameters - all_parameters}')
                if self.PARAMETER_DESCRIPTION:
                    api['parameter_description'] = self.PARAMETER_DESCRIPTION
                api_name = api['name'] if self.action.is_toolkit else 'run'
                self._api2param[api_name] = api['parameters']
                self._api2required[api_name] = api['required']

    def parse_inputs(self,
                     inputs: Union[str, dict],
                     name: str = 'run') -> dict:
        if not isinstance(inputs, dict):
            try:
                match = re.search(r'^\s*(```json\n)?(.*)\n```\s*$', inputs, re.S)
                if match:
                    inputs = match.group(2).strip()
                inputs = json.loads(inputs)
            except json.JSONDecodeError as exc:
                raise Exception(f'invalid json format: {inputs}') from exc

        input_keys = set(inputs)
        all_keys = {param['name'] for param in self._api2param[name]}
        if not input_keys.issubset(all_keys):
            raise Exception(f'unknown arguments: {input_keys - all_keys}')
        required_keys = set(self._api2required[name])
        if not input_keys.issuperset(required_keys):
            raise Exception(
                f'missing required arguments: {required_keys - input_keys}')
        return inputs


class BaseAction(metaclass=ToolMeta):
    """Base class for all actions.

    Args:
        description (:class:`Optional[dict]`): The description of the action.
            Defaults to ``None``.
        parser (:class:`Type[BaseParser]`): The parser class to process the
            action's inputs and outputs. Defaults to :class:`JsonParser`.

   提供一个标准化的工具类模板，用于实现具体的工具功能。

    核心功能：
        定义了工具的基本行为（例如，通过 run 方法执行工具逻辑）。
        支持动态调用（通过 __call__ 方法）。
        提供描述信息、输入解析和异常处理机制。
        允许开发者实现简单工具（如 run 方法）或复杂工具包（多API工具）。

    适用场景：
        定义具体工具类时的基类。
        可扩展为单一功能工具（如加法器）或多功能工具包（如计算器）。

    """

    def __init__(
            self,
            description: Optional[dict] = None,
            parser=JsonParser,
    ):
        self._description = deepcopy(description or self.__tool_description__)
        self._name = self._description['name']
        self._parser = parser(self)

    def __call__(self, inputs: str, name='run') -> ActionReturn:
        fallback_args = {'inputs': inputs, 'name': name}
        if not hasattr(self, name):
            return ActionReturn(
                fallback_args,
                type=self.name,
                errmsg=f'invalid API: {name}',
                state=ActionStatusCode.API_ERROR)
        try:
            inputs = self._parser.parse_inputs(inputs, name)
        except Exception as exc:
            return ActionReturn(
                fallback_args,
                type=self.name,
                errmsg=str(exc),
                state=ActionStatusCode.ARGS_ERROR)

        try:
            outputs = getattr(self, name)(**inputs)
        except Exception as exc:
            return ActionReturn(
                inputs,
                type=self.name,
                errmsg=str(exc),
                state=ActionStatusCode.API_ERROR)

        if isinstance(outputs, ActionReturn):
            action_return = outputs
            if not action_return.args:
                action_return.args = inputs
            if not action_return.type:
                action_return.type = self.name
        else:
            result = self._parser.parse_outputs(outputs)
            action_return = ActionReturn(inputs, type=self.name, result=result)
        return action_return

    @property
    def name(self):
        return self._name

    @property
    def is_toolkit(self):
        return self._is_toolkit

    @property
    def description(self) -> dict:
        """Description of the tool."""
        return self._description

    def __repr__(self):
        return f'{self.description}'

    __str__ = __repr__


if __name__ == "__main__":
    class Bold(BaseAction):
        '''Make text bold'''

        def run(self, text: str):
            '''
            Args:
                text (str): input text

            Returns:
                str: bold text
            '''
            return '**' + text + '**'


    action = Bold()
    print(action("Hello"))
