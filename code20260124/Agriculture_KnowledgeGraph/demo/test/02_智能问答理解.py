from demo.demo.question_answering import inner_question_answering

if __name__ == '__main__':
    # NOTE: 实现逻辑：首先判断问题所属类别 ---> 每个类别采用一套代码逻辑进行回复
    # TODO: 首先判断问题所属类别  --> 意图识别；部分逻辑代码中的信息抽取 --> 实体识别、文本生成(限制性的)
    question = "长沙适合种什么植物？"
    result = inner_question_answering(question)
    print(result)
