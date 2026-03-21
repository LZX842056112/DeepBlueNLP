from fastapi import Body
from langchain_core.messages import SystemMessage
from sse_starlette.sse import EventSourceResponse
from configs import LLM_MODELS, TEMPERATURE, logger
from server.utils import wrap_done, get_ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable
import asyncio
import json
from langchain.prompts.chat import ChatPromptTemplate
from typing import List, Optional, Union
from server.chat.utils import History
from langchain.prompts import PromptTemplate
from server.utils import get_prompt_template
from server.memory.conversation_db_buffer_memory import ConversationBufferDBMemory
from server.db.repository import add_message_to_db
from server.callback_handler.conversation_callback_handler import ConversationCallbackHandler


async def chat(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
               conversation_id: str = Body("", description="对话框ID"),
               history_len: int = Body(-1, description="从数据库中取历史消息的数量"),
               history: Union[int, List[History]] = Body([],
                                                         description="历史对话，设为一个整数可以从数据库中读取历史消息",
                                                         examples=[[
                                                             {"role": "user",
                                                              "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                             {"role": "assistant", "content": "虎头虎脑"}]]
                                                         ),
               stream: bool = Body(False, description="流式输出"),
               model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
               temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
               max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
               # top_p: float = Body(TOP_P, description="LLM 核采样。勿与temperature同时设置", gt=0.0, lt=1.0),
               prompt_name: str = Body("default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
               ):
    logger.info(
        f"Chat入参为: query={query} \n "
        f"conversation_id={conversation_id} \n "
        f"history_len={history_len} \n "
        f"history={history} \n "
        f"stream={stream} \n "
        f"model_name={model_name} \n "
        f"temperature={temperature} \n "
        f"max_tokens={max_tokens} \n "
        f"prompt_name={prompt_name} \n "
    )

    async def chat_iterator() -> AsyncIterable[str]:
        nonlocal history, max_tokens
        callback = AsyncIteratorCallbackHandler()
        callbacks = [callback]
        memory = None

        # 负责保存llm response到message db
        message_id = add_message_to_db(
            chat_type="llm_chat", query=query, conversation_id=conversation_id
        )
        conversation_callback = ConversationCallbackHandler(
            conversation_id=conversation_id,
            message_id=message_id,
            chat_type="llm_chat",
            query=query
        )
        callbacks.append(conversation_callback)

        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        # 获取支持OpenAI-API接口的模型client对象
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
        )
        logger.info(f"\n获取得到当前'模型'对象:{model}")

        if prompt_name == 'ner':
            print("不使用历史信息....")
            global_messages = [
                {'role': 'user', 'content': '```今天还有去北京的飞机票吗```'},
                {
                    'role': 'assistant',
                    'content': '[{"entity_type":"出发时间","entity_span":"今天"},{"entity_type":"交通方式","entity_span":"飞机"},{"entity_type":"目的地","entity_span":"北京"}]'
                }
            ]
            history = [History.from_data(h) for h in global_messages]
            prompt_template = get_prompt_template("llm_chat", prompt_name)
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            system = '从当前用户给定的三个`包含的文本中提取实体单词，不允许从其它位置进行提取。\n实体类别列表为:["出发时间", "出发地", "目的地", "交通方式"]\n并以json格式的结果返回，json中包含实体类型和实体片段两个字段，不允许返回其它内容。\n比如返回的格式为:[{"entity_type":"出发时间","entity_span":"明天"},...]\n'
            system_msg = SystemMessage(content=system)
            chat_prompt = ChatPromptTemplate.from_messages(
                [system_msg] +
                [i.to_msg_template() for i in history] + [input_msg]
            )
        else:
            if history:  # 优先使用调用方传入的历史消息
                history = [History.from_data(h) for h in history]
                prompt_template = get_prompt_template("llm_chat", prompt_name)
                logger.info(f"\n模版字符串为:{prompt_template}")
                input_msg = History(role="user", content=prompt_template).to_msg_template(False)
                chat_prompt = ChatPromptTemplate.from_messages(
                    [i.to_msg_template() for i in history] + [input_msg])
            elif conversation_id and history_len > 0:  # 前端要求从数据库取历史消息
                # 使用memory 时必须 prompt 必须含有memory.memory_key 对应的变量
                prompt = get_prompt_template("llm_chat", "with_history")
                chat_prompt = PromptTemplate.from_template(prompt)
                # 根据conversation_id 获取message 列表进而拼凑 memory
                memory = ConversationBufferDBMemory(conversation_id=conversation_id,
                                                    llm=model,
                                                    message_limit=history_len)
            else:
                prompt_template = get_prompt_template("llm_chat", prompt_name)
                input_msg = History(role="user", content=prompt_template).to_msg_template(False)
                chat_prompt = ChatPromptTemplate.from_messages([input_msg])
        logger.info(f"\n模型模版对象为:{chat_prompt}")

        chain = LLMChain(prompt=chat_prompt, llm=model, memory=memory)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"input": query}),
            callback.done),
        )

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps(
                    {"text": token, "message_id": message_id},
                    ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps(
                {"text": answer, "message_id": message_id},
                ensure_ascii=False)

        await task

    return EventSourceResponse(chat_iterator())
