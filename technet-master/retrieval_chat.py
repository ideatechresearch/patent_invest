# 引导ChatGPT先生成问题，再生成答案，得到（金融文本、问题、答案）这样的三元组，并进一步使用阅读理解任务模板改写为指令对
# 为了增加数据的数量和多样性，我们通过小样本思维链提示（Few-shot Chain-of-Thought Prompting）方法，让ChatGPT在提示词的引导下，根据种子任务生成超过5万个新问答对，其中的答案也带有插件命令
#检索增强指令的构造分为三步。第一步，我们根据新闻和研报等金融文本构造金融分析问题。第二步，我们在知识库中检索与问题有关的文档，其中参考文档源于我们构建金融知识库，包含18k研报和69k金融新闻。第三步，我们将问题和参考资料结合在一起，生成问题的答案。在这个过程中，问题和答案是由ChatGPT通过Chain-of-Retrieval (CoR) prompting方法生成的。最终我们构建了一个由20k条检索增强指令组成的数据集，其中的指令涵盖了金融领域中主要的分析形式，包括行业分析、政策分析、投资建议、公司战略规划等。
# "instruction": "作为一名金融领域专家，请回答下面的问题。下面的材料仅供参考。\n（注意：1、材料可能与问题无关，请忽略无关材料，并基于已有知识回答问题。 2、尽量不要直接复制材料内容作为答案，而应该将材料内容作为事件的补充与潜在分析，启发思考）。3、请直接给出分析和答案，无需给出具体参考了哪篇文档）；\n参考材料 1:\n《【机会掘金】新能源汽车再获政策支持 市场渗透率仍将持续攀升》\n
# \n\n 问题：

# "instruction":"请根据以下提供的上下文回答相应问题: \n上下文:\n
# "instruction": "请根据此文本，提取出文本的关键词。\n下面给出了三个样例，按照此样例输出最后一个文本的答案。\n文本：统计文本中单词\"the\"出现的次数。猫坐在房间角落的垫子上。\n答案：关键字：统计、文本、单词
# "instruction": "请找出下文中的实体并返回实体的类型，共有以下几类实体类型：事件类型、时间、主体，如果事件没有相对应的时间/主体/数值，则输出\"无时间\"/\"无主体\"/\"无数值\"\n文本：
# "instruction": "请根据上下文，从文本中提取表达的情绪，选项为积极、消极，请在这两个选项中选出唯一正确的选项。请遵循以下示例，仅输出情绪类别。\n\n上下文：

# "instruction": "上下文：    。 \n请根据上下文回答下面的问题：


# "question": "如何化解当前经济的主要矛盾？",
# "question": "为什么宁可期权价值归零也不提早割肉？",
#         "reference": [

# zero_shot_prompts = [
# few_shot_prompts = [

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.generation.utils import GenerationConfig
def extract_questions_and_text(input_string):
    # 使用问号来分割字符串为问题列表
    questions_list = input_string.split("？")

    # 第一个问题及之前的所有问题合并为一个字符串
    first_content = "？".join(questions_list[:-1]) + "？"

    # 最后一个问题后面的文本内容作为第二个内容
    second_content = questions_list[-1]

    return first_content, second_content

Tools = [
    # 工具1 获取当前时刻的时间
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "当你想知道现在的时间时非常有用。",
            "parameters": {}  # 因为获取当前时间无需输入参数，因此parameters为空字典
        }
    },
    # 工具2 获取指定城市的天气
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "当你想查询指定城市的天气时非常有用。",
            "parameters": {  # 查询天气时需要提供位置，因此参数设置为location
                        "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市或县区，比如北京市、杭州市、余杭区等。"
                    }
                }
            },
            "required": [
                "location"
            ]
        }
    }
]
