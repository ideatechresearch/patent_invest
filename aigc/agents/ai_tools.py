AI_Tools = [
    {
        "type": "function",
        "function": {
            "name": "get_times_shift",
            "description": "当你想知道时间时非常有用。获取当前时间，并根据偏移的天数和小时数调整时间。",
            "parameters": {
                "type": "object",
                "properties": {
                    "days_shift": {
                        "type": "integer",
                        "description": "偏移的天数，>0 表示未来，<0 表示过去，0 表示当前日期。",
                        "default": 0
                    },
                    "hours_shift": {
                        "type": "integer",
                        "description": "偏移的小时数，>0 表示未来，<0 表示过去，0 表示当前时间。",
                        "default": 0
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "date_range_calculator",
            "description": "计算基于参考日期的时间范围，支持按天、周、月、季度、年或半年计算日期范围，支持偏移周期数和返回多个周期范围。",
            "parameters": {
                "type": "object",
                "properties": {
                    "period_type": {
                        "type": "string",
                        "enum": ["days", "weeks", "months", "quarters", "year", "half_year"],
                        "description": "时间周期类型，可以是 'days'、'weeks'、'months'、'quarters'、'year' 或 'half_year'。"
                    },
                    "date": {
                        "type": "string",
                        "format": "date",
                        "description": "基准日期，格式为 'YYYY-MM-DD'，默认为当前日期。"
                    },
                    "shift": {
                        "type": "integer",
                        "default": 0,
                        "description": "偏移量，表示时间周期的偏移，0 表示当前周期，负值表示过去，正值表示未来。"
                    },
                    "count": {
                        "type": "integer",
                        "default": 1,
                        "description": "时间周期数量，表示返回多少个周期的日期范围,默认为 1。"
                    }
                },
                "required": ["period_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search_async",
            "description": "通过提供的查询文本执行网络搜索，外部知识实时动态信息检索。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "需要在网络上搜索的查询文本。",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "用于访问网络搜索工具的API密钥，默认不需要提供。",
                        "default": "Config.GLM_Service_Key"
                    }
                },
                "required": ["text"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_bmap_location",
            "description": "使用百度地图API搜索指定地点。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索地点的关键词。",
                    },
                    "region": {
                        "type": "string",
                        "description": "限制搜索的区域，例如城市名。",
                        "default": ""
                    },
                    "limit": {
                        "type": "boolean",
                        "description": "是否仅限指定区域内搜索，默认为 True。",
                        "default": True
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "auto_translate",
            "description": "自动翻译工具，根据输入的文本和指定的翻译模型完成语言检测与翻译。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "需要翻译的文本内容。"
                    },
                    "model_name": {
                        "type": "string",
                        "description": "指定使用的翻译模型，支持 'baidu','tencent','xunfei'。其他模型(如:'qwen','moonshot','baichuan',doubao','hunyuan','spark')将自动调用 AI 翻译,将提供目标语言释义和例句.",
                        "default": "baidu"
                    },
                    "source": {
                        "type": "string",
                        "description": "原文本的语言代码，支持具体语言代码（如 'en', 'zh','zh-TW','ja','ru','it','fr','pt','th','ko','es','vi','id'）或 'auto'（自动检测）。",
                        "default": "auto"
                    },
                    "target": {
                        "type": "string",
                        "description": "目标翻译语言代码，支持具体语言代码（如 'en', 'zh','zh-TW','ja','ru','it','fr','pt','th','ko','es','vi','id'）或 'auto'（根据源语言自动设定）。",
                        "default": "auto"
                    }
                },
                "required": ["text"]
            }
        }
    }
]


'''
功能型 (function): 核心逻辑的实现。
工具型 (tool): 通过现成工具提供结果。
API型 (api): 封装外部接口调用。
服务型 (service): 长时间运行或后台任务。
插件型 (plugin): 为现有系统提供扩展。
数据型 (data): 处理和生成数据的任务。
查询型 (query): 面向数据检索的操作。
'''
