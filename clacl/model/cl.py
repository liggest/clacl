
from __future__ import annotations
from typing import Callable, TYPE_CHECKING
from collections import UserDict, defaultdict
from enum import Enum
from pathlib import Path

import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig

from clacl.util import logger

if TYPE_CHECKING:
    from typing_extensions import Self

class AdapterState(str, Enum):
    Missing = "missing"
    Freeze = "freeze"
    TuneOnce = "tune_once"
    TuneAll = "tune_all"
    CL = "cl"

class CLAdapter(nn.Module):            

    def __init__(self, factory: Callable[[str, PretrainedConfig], nn.Module], state=AdapterState.CL):
        super().__init__()
        self.adapters = self._init_adapters()
        self._current_task: str | None = None
        self.module_factory = factory
        self.state = state
        self._previous_task: str | None = None

    def _init_adapters(self):
        return nn.ModuleDict()

    @property
    def current_task(self):
        return self._current_task
    
    @current_task.setter
    def current_task(self, val: str | None):
        self._previous_task = self._current_task
        self._current_task = val

    def add_adapter(self, task_name: str, config: PretrainedConfig):
        if (self.state == AdapterState.TuneOnce and self.current_task):
            self.set_grad(self.current_task, freeze=True)
        
        if (not_special_task(task_name)
                and self.adapters 
                and self.state != AdapterState.CL):
            return  # only one module

        adapter = self.module_factory(task_name, config)
        self.set_adapter(task_name, adapter)

        if (self.state == AdapterState.Freeze):
            self.set_grad(task_name, freeze=True)

    def set_adapter(self, task_name: str, adapter: nn.Module):
        if (not_special_task(task_name)
                and self.adapters 
                and self.state != AdapterState.CL 
                and task_name not in self.adapters):
            return  # only one module
        
        # self.previous_task = self.current_task
        self.adapters[task_name] = adapter
        self.current_task = task_name

    def set_grad(self, task_name: str | None = None, freeze=True):
        if task_name is None:
            task_name = self.current_task
        assert task_name
        self.adapters[task_name].requires_grad_(not freeze)

    def forward(self, hidden_states):
        if not self.current_task:
            logger.warning(f"no {self.current_task = !r} in {self!r}, do nothing to the input")
            return hidden_states
        return self.adapters[self.current_task](hidden_states)
    
    def average_adapter(self, config: PretrainedConfig):
        task_name = "_average_"
        if self.current_task == task_name:
            return
        if self.state != AdapterState.CL:
            return
        if (task_name not in self.adapters):
            self.add_adapter(task_name, config)  # will also change current task
        else:
            self.set_adapter(task_name, self.adapters[task_name])
        self.set_grad(task_name, freeze=True)
        average = self.adapters[task_name]
        if self._previous_task:
            previous_device = next(self.adapters[self._previous_task].parameters()).device
            average.to(previous_device)
        average_state = average.state_dict()
        states = [adapter.state_dict() for t_name, adapter in self.adapters.items() if not_special_task(t_name)]
        n = len(states)
        for key in average_state:
            average_state[key] = sum(state[key] for state in states) / n
        average.load_state_dict(average_state)

    _default_group = "adapters"
    def manage_by(self, manager: CLManager | str | None = None, group: str | None = None, name: str | None = None):
        if not isinstance(manager, CLManager):
            manager = CLManager.of(manager)
        if group is None:
            group = self._default_group
        name, module = manager.add(self, group, name)
        self._manager = manager._name
        self._group = group
        self._name = name
        return module

class CLParameter(nn.Module):

    def __init__(self, factory: Callable[[str, PretrainedConfig], nn.Parameter], state=AdapterState.CL):
        super().__init__() 
        self.adapters = self._init_adapters()
        self._current_task: str | None = None
        self.module_factory = factory
        self.state = state
        self._previous_task: str | None = None

    def _init_adapters(self):
        return nn.ParameterDict()

    current_task = CLAdapter.current_task

    add_adapter = CLAdapter.add_adapter
    set_adapter = CLAdapter.set_adapter
    set_grad = CLAdapter.set_grad

    def forward(self) -> nn.Parameter:
        assert self.current_task, f"{self.current_task = !r} in {self!r}"
        return self.adapters[self.current_task]

    def average_adapter(self, config: PretrainedConfig):
        task_name = "_average_"
        if self.current_task == task_name:
            return
        if self.state != AdapterState.CL:
            return
        if (task_name not in self.adapters):
            self.add_adapter(task_name, config)  # will also change current task
        else:
            self.set_adapter(task_name, self.adapters[task_name])
        self.set_grad(task_name, freeze=True)
        average: nn.Parameter = self.adapters[task_name]
        if self._previous_task:
            previous: nn.Parameter = self.adapters[self._previous_task]
            average.to(previous.device)
        tensors = [adapter.data for t_name, adapter in self.adapters.items() if not_special_task(t_name)]
        n = len(tensors)
        average.data = sum(tensors) / n

    _default_group = "parameters"

    manage_by = CLAdapter.manage_by

CLModule = CLAdapter | CLParameter

def not_special_task(task_name: str):
    return not task_name.startswith("_")

def dump_cl_module_state(module: CLModule):
    return {
        "class": module.__class__.__name__,
        "state_dict": module.state_dict(),
        "current_task": module.current_task,
        "adapter_state": module.state.value,
    }

def load_cl_module_state(state: dict, manager: CLManager | None = None, group: str | None = None, name: str | None = None):
    cls: type[CLModule] = {
        CLAdapter.__name__: CLAdapter,
        CLParameter.__name__: CLParameter,
    }.get(state["class"])
    assert cls
    adapter_state = AdapterState(state["adapter_state"])

    def missing_factory(task_name: str, config: PretrainedConfig):
        raise NotImplementedError
    
    module = cls(missing_factory, adapter_state)
    module.current_task = state["current_task"]
    module.load_state_dict(state["state_dict"])

    if manager:
        module._manager = manager._name
    if group:
        module._group = group
    if name:
        module._name = name

    return module


class CLManager(UserDict[str, dict[str, CLModule]]):

    _default_manager_name = "model"
    _managers: dict[str, "Self"] = {}

    @classmethod
    def of(cls, manager_name: str | None = None):
        if manager_name is None:
            manager_name = cls._default_manager_name
        if (manager := cls._managers.get(manager_name)) is not None:
            return manager
        return cls(manager_name)

    def __init__(self, name: str | None = None):
        # super().__init__()
        self.data = defaultdict(dict)  # group_name : group_dict
        if name is None:
            name = self._default_manager_name
        assert name not in self._managers
        self._name = name
        self._managers[name] = self

    def add(self, module: CLModule, group: str, name: str | None):
        group_dict = self.data[group]
        if name is None:
            name = str(len(group_dict))
        assert name not in group_dict
        group_dict[name] = module
        return name, module

    def named_module_gen(self, f: Callable[[CLModule, str, str], bool] | None = None):
        for group_name, group_dict in self.data.items():
            for name, module in group_dict.items():
                if f is None or f(module, group_name, name):
                    yield group_name, name, module

    def module_gen(self, f: Callable[[CLModule, str, str], bool] | None = None):
        for *_, module in self.named_module_gen(f):
            yield module

    def set_task(self, task_name: str, f: Callable[[CLModule, str, str], bool] | None = None):
        non_set = 0
        for module in self.module_gen(f):
            if task_name in module.adapters:
                module.current_task = task_name
            else:
                non_set += 1
        if non_set:
            print(f"{task_name} not exists in {non_set} modules. Unable to set {task_name} for them.")

    def set_task_back(self, f: Callable[[CLModule, str, str], bool] | None = None):
        for module in self.module_gen(f):
            module.current_task = module._previous_task
    
    def set_task_grad(self, task_name: str | None = None, freeze=True, f: Callable[[CLModule, str, str], bool] | None = None):
        for module in self.module_gen(f):
            module.set_grad(task_name, freeze)

    def save(self, path: Path):
        torch.save({
            "name": self._name,
            "data": {
                group_name : {
                    name: dump_cl_module_state(module)
                    for name, module in group_dict.items()
                }
                for group_name, group_dict in self.data.items()
            }
        }, path)

    @classmethod
    def load(cls, path: Path):
        # This can load CLManager but not the model
        _data = torch.load(path, weights_only=True)
        manager = cls.of(_data["name"])
        manager.data.update({
            {
                group_name : {
                    name: load_cl_module_state(state, manager, group_name, name)
                    for name, state in group_dict.items()
                }
                for group_name, group_dict in _data["data"].items()
            }
        })
        return manager
