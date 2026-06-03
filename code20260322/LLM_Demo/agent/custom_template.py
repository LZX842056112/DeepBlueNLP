import json
from typing import List

from langchain_classic.agents import AgentOutputParser
from langchain_core.agents import AgentFinish, AgentAction
from langchain_core.prompts import StringPromptTemplate
from langchain_core.tools import Tool


class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts

        def _fetch_tool_input_schema(_tool):
            if _tool.args_schema is None:
                return {}
            _schema = {}
            _fields = _tool.args_schema.model_fields
            for _field_key, _field in _fields.items():
                _schema[_field_key] = _field.description
            return _schema

        kwargs["tools"] = "\n\n".join(
            [f"{tool.name}: {tool.description} ; input args: {_fetch_tool_input_schema(tool)}" for tool in self.tools])
        # kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])

        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):
    begin: bool = False

    def __init__(self):
        super().__init__()
        self.begin = True

    def parse(self, llm_output: str) -> AgentFinish | tuple[dict[str, str], str] | AgentAction:
        print(f"模型原始的输出文本:{llm_output}\n\n")
        llm_output = llm_output.strip()
        if llm_output.startswith("Thought: "):
            llm_output = llm_output[len("Thought: "):]

        if self.begin:
            self.begin = False
            stop_words = ["Observation:"]
            min_index = len(llm_output)
            for stop_word in stop_words:
                index = llm_output.find(stop_word)
                if index != -1 and index < min_index:
                    min_index = index
                llm_output = llm_output[:min_index]

        if "Final Answer:" in llm_output:
            self.begin = True
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:", 1)[-1].strip()},
                log=llm_output,
            )
        parts = llm_output.split("Action:")
        if len(parts) < 2:
            return AgentFinish(
                return_values={"output": f"调用agent工具失败，该回答为大模型自身能力的回答:\n\n `{llm_output}`"},
                log=llm_output,
            )

        action = parts[1].split("Action Input:")[0].strip()
        action_input = parts[1].split("Action Input:")[1].strip()
        try:
            print(f"调用的agent:{action}")
            print(f"调用的agent入参:{action_input}")
            try:
                action_input = json.loads(action_input)
            except Exception:
                try:
                    action_input = eval(action_input)
                except:
                    try:
                        sid = action_input.find("{")
                        eid = action_input.rfind("}")
                        action_input = json.loads(action_input[sid:eid + 1])
                    except:
                        action_input = action_input.strip(" ").strip('"')

            ans = AgentAction(
                tool=action,
                tool_input=action_input,
                log=llm_output
            )
            return ans
        except:
            return AgentFinish(
                return_values={"output": f"调用agent失败: `{llm_output}`"},
                log=llm_output,
            )
