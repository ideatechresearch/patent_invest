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
            "description": "通过提供的查询文本执行网络搜索。比如查看新闻、最新消息、价格走势等实时动态，外部知识信息检索。",
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

AI_Tools_AUTO = [{'type': 'function', 'function': {'name': 'get_times_shift',
                                                   'description': '当你想知道时间时非常有用。获取当前时间，并根据偏移的天数和小时数调整时间。',
                                                   'parameters': {'type': 'object', 'properties': {
                                                       'days_shift': {'type': 'integer',
                                                                      'description': '偏移的天数，>0 表示未来，<0 表示过去，0 表示当前日期。',
                                                                      'default': 0}, 'hours_shift': {'type': 'integer',
                                                                                                     'description': '偏移的小时数，>0 表示未来，<0 表示过去，0 表示当前时间。',
                                                                                                     'default': 0}},
                                                                  'required': []}}}, {'type': 'function',
                                                                                      'function': {
                                                                                          'name': 'date_range_calculator',
                                                                                          'description': '计算基于参考日期的时间范围，支持按天、周、月、季度、年或半年计算日期范围，支持偏移周期数和返回多个周期范围。',
                                                                                          'parameters': {
                                                                                              'type': 'object',
                                                                                              'properties': {
                                                                                                  'period_type': {
                                                                                                      'type': 'string',
                                                                                                      'enum': [
                                                                                                          'days',
                                                                                                          'weeks',
                                                                                                          'months',
                                                                                                          'quarters',
                                                                                                          'year',
                                                                                                          'half_year'],
                                                                                                      'description': "时间周期类型，可以是 'days'、'weeks'、'months'、'quarters'、'year' 或 'half_year'。"},
                                                                                                  'date': {
                                                                                                      'type': 'string',
                                                                                                      'format': 'date',
                                                                                                      'description': "基准日期，格式为 'YYYY-MM-DD'，默认为当前日期。"},
                                                                                                  'shift': {
                                                                                                      'type': 'integer',
                                                                                                      'default': 0,
                                                                                                      'description': '偏移量，表示时间周期的偏移，0 表示当前周期，负值表示过去，正值表示未来。'},
                                                                                                  'count': {
                                                                                                      'type': 'integer',
                                                                                                      'default': 1,
                                                                                                      'description': '时间周期数量，表示返回多少个周期的日期范围,默认为 1。'}},
                                                                                              'required': [
                                                                                                  'period_type']}}},
                 {'type': 'function', 'function': {'name': 'web_search_async',
                                                   'description': '通过提供的查询文本执行网络搜索。比如查看新闻、最新消息、价格走势等实时动态，外部知识信息检索。',
                                                   'parameters': {'type': 'object', 'properties': {
                                                       'text': {'type': 'string',
                                                                'description': '需要在网络上搜索的查询文本。'},
                                                       'api_key': {'type': 'string',
                                                                   'description': '用于访问网络搜索工具的API密钥，默认不需要提供。',
                                                                   'default': 'Config.GLM_Service_Key'}},
                                                                  'required': ['text']}}},
                 {'type': 'function',
                  'function': {'name': 'search_bmap_location', 'description': '使用百度地图API搜索指定地点。',
                               'parameters': {'type': 'object', 'properties': {
                                   'query': {'type': 'string', 'description': '搜索地点的关键词。'},
                                   'region': {'type': 'string', 'description': '限制搜索的区域，例如城市名。',
                                              'default': ''}, 'limit': {'type': 'boolean',
                                                                        'description': '是否仅限指定区域内搜索，默认为 True。',
                                                                        'default': True}},
                                              'required': ['query']}}}, {'type': 'function',
                                                                         'function': {'name': 'auto_translate',
                                                                                      'description': '自动翻译工具，根据输入的文本和指定的翻译模型完成语言检测与翻译。',
                                                                                      'parameters': {
                                                                                          'type': 'object',
                                                                                          'properties': {
                                                                                              'text': {
                                                                                                  'type': 'string',
                                                                                                  'description': '需要翻译的文本内容。'},
                                                                                              'model_name': {
                                                                                                  'type': 'string',
                                                                                                  'description': "指定使用的翻译模型，支持 'baidu','tencent','xunfei'。其他模型(如:'qwen','moonshot','baichuan',doubao','hunyuan','spark')将自动调用 AI 翻译,将提供目标语言释义和例句.",
                                                                                                  'default': 'baidu'},
                                                                                              'source': {
                                                                                                  'type': 'string',
                                                                                                  'description': "原文本的语言代码，支持具体语言代码（如 'en', 'zh','zh-TW','ja','ru','it','fr','pt','th','ko','es','vi','id'）或 'auto'（自动检测）。",
                                                                                                  'default': 'auto'},
                                                                                              'target': {
                                                                                                  'type': 'string',
                                                                                                  'description': "目标翻译语言代码，支持具体语言代码（如 'en', 'zh','zh-TW','ja','ru','it','fr','pt','th','ko','es','vi','id'）或 'auto'（根据源语言自动设定）。",
                                                                                                  'default': 'auto'}},
                                                                                          'required': [
                                                                                              'text']}}},
                 {'type': 'function', 'function': {'name': 'get_times_shift',
                                                   'description': '该函数用于获取相对于当前时间偏移指定天数和小时数后的时间，并返回格式化后的时间字符串。',
                                                   'parameters': {'type': 'object', 'properties': {
                                                       'days_shift': {'type': 'integer',
                                                                      'description': '偏移的天数，>0 表示未来，<0 表示过去，0 表示当前日期。',
                                                                      'default': 0}, 'hours_shift': {'type': 'integer',
                                                                                                     'description': '偏移的小时数，>0 表示未来，<0 表示过去，0 表示当前时间。',
                                                                                                     'default': 0}},
                                                                  'required': []}}}, {'type': 'function',
                                                                                      'function': {
                                                                                          'name': 'date_range_calculator',
                                                                                          'description': '计算基于参考日期的时间范围。',
                                                                                          'parameters': {
                                                                                              'type': 'object',
                                                                                              'properties': {
                                                                                                  'period_type': {
                                                                                                      'type': 'string',
                                                                                                      'description': "时间周期类型，'days'、'weeks'、'months' 等"},
                                                                                                  'date': {
                                                                                                      'type': 'string',
                                                                                                      'description': "基准日期，格式为 'YYYY-MM-DD'",
                                                                                                      'default': None},
                                                                                                  'shift': {
                                                                                                      'type': 'integer',
                                                                                                      'description': '半年偏移量，0 表示当前半年，-1 表示前一半年，1 表示下一半年。',
                                                                                                      'default': 0},
                                                                                                  'count': {
                                                                                                      'type': 'integer',
                                                                                                      'description': '时间周期数量，表示从参考日期向前或向后的时长',
                                                                                                      'default': 1}},
                                                                                              'required': [
                                                                                                  'period_type']}}},
                 {'type': 'function',
                  'function': {'name': 'get_day_range', 'description': '根据给定的日期、偏移量和计数返回日期范围。',
                               'parameters': {'type': 'object', 'properties': {
                                   'date': {'type': 'string',
                                            'description': '起始日期，可以是字符串格式，如果未提供则使用当前日期。',
                                            'default': None},
                                   'shift': {'type': 'integer', 'description': '相对于起始日期的天数偏移量，默认为0。',
                                             'default': 0},
                                   'count': {'type': 'integer', 'description': '从起始日期开始的天数范围，默认为1。',
                                             'default': 1}},
                                              'required': []}}},
                 {'type': 'function', 'function': {'name': 'get_week_range',
                                                   'description': '获取指定日期所在周的开始和结束日期。支持通过 shift 参数偏移周。',
                                                   'parameters': {'type': 'object',
                                                                  'properties': {
                                                                      'date': {
                                                                          'type': 'string',
                                                                          'description': '指定的日期（默认为当前日期）。',
                                                                          'default': None},
                                                                      'shift': {
                                                                          'type': 'integer',
                                                                          'description': '偏移周数，>0 表示未来的周，<0 表示过去的周，0 表示当前周。',
                                                                          'default': 0},
                                                                      'count': {
                                                                          'type': 'integer',
                                                                          'description': '控制返回的周数范围，默认为 1，表示返回一个周的日期范围。',
                                                                          'default': 1}},
                                                                  'required': []}}},
                 {'type': 'function', 'function': {'name': 'get_month_range',
                                                   'description': '获取指定日期所在月的开始和结束日期。支持通过 shift 参数偏移月数，和通过 count 控制返回的月份范围。',
                                                   'parameters': {'type': 'object', 'properties': {
                                                       'date': {'type': 'string',
                                                                'description': '指定的日期（默认为当前日期）。',
                                                                'default': None}, 'shift': {'type': 'integer',
                                                                                            'description': '偏移的月数，>0 表示未来的月，<0 表示过去的月，0 表示当前月。',
                                                                                            'default': 0},
                                                       'count': {'type': 'integer',
                                                                 'description': '控制返回的月份范围，默认为 1，表示返回一个月的开始和结束日期。',
                                                                 'default': 1}}, 'required': []}}}, {'type': 'function',
                                                                                                     'function': {
                                                                                                         'name': 'get_quarter_range',
                                                                                                         'description': '获取指定日期所在季度的开始和结束日期。支持通过 shift 参数偏移季度数，和通过 count 控制返回的季度范围。',
                                                                                                         'parameters': {
                                                                                                             'type': 'object',
                                                                                                             'properties': {
                                                                                                                 'date': {
                                                                                                                     'type': 'string',
                                                                                                                     'description': '指定的日期（默认为当前日期）。',
                                                                                                                     'default': None},
                                                                                                                 'shift': {
                                                                                                                     'type': 'integer',
                                                                                                                     'description': '偏移的季度数，>0 表示未来的季度，<0 表示过去的季度，0 表示当前季度。',
                                                                                                                     'default': 0},
                                                                                                                 'count': {
                                                                                                                     'type': 'integer',
                                                                                                                     'description': '控制返回的季度范围，默认为 1，表示返回一个季度的开始和结束日期。',
                                                                                                                     'default': 1}},
                                                                                                             'required': []}}},
                 {'type': 'function', 'function': {'name': 'get_year_range',
                                                   'description': '获取指定日期所在年的开始和结束日期。支持通过 shift 参数偏移年数，和通过 count 控制返回的年度范围。',
                                                   'parameters': {'type': 'object', 'properties': {
                                                       'date': {'type': 'string',
                                                                'description': '指定的日期（默认为当前日期）。',
                                                                'default': None}, 'shift': {'type': 'integer',
                                                                                            'description': '偏移的年数，>0 表示未来的年，<0 表示过去的年，0 表示当前年。',
                                                                                            'default': 0},
                                                       'count': {'type': 'integer',
                                                                 'description': '控制返回的年度范围，默认为 1，表示返回一年的开始和结束日期。',
                                                                 'default': 1}}, 'required': []}}}, {'type': 'function',
                                                                                                     'function': {
                                                                                                         'name': 'get_half_year_range',
                                                                                                         'description': '获取指定日期所在的半年（前半年或后半年）范围。支持通过 shift 参数偏移半年数，和通过 count 控制返回的半年数范围。',
                                                                                                         'parameters': {
                                                                                                             'type': 'object',
                                                                                                             'properties': {
                                                                                                                 'date': {
                                                                                                                     'type': 'string',
                                                                                                                     'description': '指定的日期（默认为当前日期）。',
                                                                                                                     'default': None},
                                                                                                                 'shift': {
                                                                                                                     'type': 'integer',
                                                                                                                     'description': '半年偏移量，0 表示当前半年，-1 表示前一半年，1 表示下一半年。',
                                                                                                                     'default': 0},
                                                                                                                 'count': {
                                                                                                                     'type': 'integer',
                                                                                                                     'description': '返回的半年范围，默认为 1，表示返回一个半年的开始和结束日期。',
                                                                                                                     'default': 1}},
                                                                                                             'required': []}}},
                 {'type': 'function', 'function': {'name': 'ideatech_knowledge',
                                                   'description': '该函数用于根据查询从知识图谱中检索相关内容，并可选地使用重排序模型对结果进行重排序。',
                                                   'parameters': {'type': 'object',
                                                                  'properties': {'query': {'type': 'string',
                                                                                           'description': '用户输入的查询字符串，用于在知识图谱中查找相关内容。'},
                                                                                 'rerank_model': {
                                                                                     'type': 'string',
                                                                                     'description': "用于重排序检索结果的模型名称，默认为 'BAAI/bge-reranker-v2-m3'。",
                                                                                     'default': 'BAAI/bge-reranker-v2-m3'},
                                                                                 'file': {'type': 'object',
                                                                                          'description': '上传的文件对象，仅当版本号为3时有效，且文件必须为PDF格式。',
                                                                                          'default': None},
                                                                                 'version': {'type': 'integer',
                                                                                             'description': '版本号，用于控制不同的处理逻辑，默认为0。',
                                                                                             'default': 0}},
                                                                  'required': ['query']}}},
                 {'type': 'function', 'function': {
                     'name': 'web_search_async',
                     'description': '异步执行网页搜索，使用给定的文本和API密钥，并返回搜索结果。',
                     'parameters': {'type': 'object',
                                    'properties': {
                                        'text': {'type': 'string', 'description': '要搜索的文本内容。', 'default': ''},
                                        'api_key': {'type': 'string', 'description': '用于授权的API密钥。',
                                                    'default': 'Config.GLM_Service_Key'},
                                        'kwargs': {'type': 'object',
                                                   'description': '额外的关键字参数，用于扩展请求数据。',
                                                   'additionalProperties': True, 'default': {}}},
                                    'required': ['text']}}},
                 {'type': 'function',
                  'function': {'name': 'get_weather', 'description': '获取指定城市的天气信息，可以指定天数或日期。',
                               'parameters': {'type': 'object', 'properties': {
                                   'city': {'type': 'string', 'description': '城市名称，用于查询该城市的天气信息。'},
                                   'days': {'type': 'integer',
                                            'description': '未来几天的天气预报，如果设置为大于0的整数，则返回相应天数的天气预报。',
                                            'default': 0},
                                   'date': {'type': 'string', 'description': '特定日期的天气预报，格式应符合API要求。',
                                            'default': None}}, 'required': ['city']}}}, {'type': 'function',
                                                                                         'function': {
                                                                                             'name': 'duckduckgo_search',
                                                                                             'description': '使用 DuckDuckGo API 进行搜索并返回结果。',
                                                                                             'parameters': {
                                                                                                 'type': 'object',
                                                                                                 'properties': {
                                                                                                     'query': {
                                                                                                         'type': 'string',
                                                                                                         'description': '搜索查询字符串。'}},
                                                                                                 'required': [
                                                                                                     'query']}}},
                 {'type': 'function', 'function': {'name': 'web_search_tavily',
                                                   'description': '使用 Tavily API 进行网页搜索，支持指定主题、时间范围、搜索深度和天数等参数。',
                                                   'parameters': {'type': 'object', 'properties': {
                                                       'text': {'type': 'string',
                                                                'description': '要搜索的文本或查询字符串。'},
                                                       'topic': {'type': 'string',
                                                                 'description': "搜索的主题，可以是 'general' 或 'news'。",
                                                                 'default': 'general'}, 'time_range': {'type': 'string',
                                                                                                       'description': "搜索的时间范围，可以是 'day', 'week', 'month', 'year', 'd', 'w', 'm', 'y'。",
                                                                                                       'default': 'month'},
                                                       'search_depth': {'type': 'string',
                                                                        'description': "搜索的深度，可以是 'basic' 或 'advanced'。",
                                                                        'default': 'basic'}, 'days': {'type': 'integer',
                                                                                                      'description': "如果主题是 'news'，则指定搜索的天数。",
                                                                                                      'default': 7},
                                                       'api_key': {'type': 'string',
                                                                   'description': 'Tavily API 的密钥。',
                                                                   'default': 'Config.TAVILY_Api_Key'},
                                                       'kwargs': {'type': 'object',
                                                                  'description': '其他可选参数，以关键字参数的形式传递。'}},
                                                                  'required': ['text']}}},
                 {'type': 'function', 'function': {
                     'name': 'web_extract_tavily', 'description': '通过 Tavily API 提取给定 URL 的信息。',
                     'parameters': {'type': 'object', 'properties': {
                         'urls': {'type': 'array', 'items': {'type': 'string'},
                                  'description': '要提取信息的 URL 列表。'},
                         'api_key': {'type': 'string', 'description': 'Tavily API 的访问密钥。',
                                     'default': 'Config.TAVILY_Api_Key'}},
                                    'required': ['urls']}}}, {'type': 'function', 'function': {'name': 'search_by_api',
                                                                                               'description': '通过指定的搜索引擎API进行搜索，支持多种搜索引擎和地理位置选项。',
                                                                                               'parameters': {
                                                                                                   'type': 'object',
                                                                                                   'properties': {
                                                                                                       'query': {
                                                                                                           'type': 'string',
                                                                                                           'description': '搜索查询字符串，用户希望搜索的内容。'},
                                                                                                       'location': {
                                                                                                           'type': 'string',
                                                                                                           'description': '搜索的位置信息，用于定位特定地区的搜索结果。',
                                                                                                           'default': None},
                                                                                                       'engine': {
                                                                                                           'type': 'string',
                                                                                                           'description': "指定使用的搜索引擎，可选值包括：'google', 'bing', 'baidu', 'naver', 'yahoo', 'youtube', 'google_videos', 'google_news', 'google_images', 'amazon_search', 'shein_search'。",
                                                                                                           'default': 'google'},
                                                                                                       'api_key': {
                                                                                                           'type': 'string',
                                                                                                           'description': '访问搜索引擎API所需的API密钥。',
                                                                                                           'default': 'TwfFtq2uyZw6WzaDXFWeETZZ'}},
                                                                                                   'required': [
                                                                                                       'query']}}},
                 {'type': 'function', 'function': {'name': 'search_bmap_location',
                                                   'description': '这是一个异步函数，用于通过百度地图API搜索指定区域内的地点建议。',
                                                   'parameters': {'type': 'object',
                                                                  'properties': {'query': {'type': 'string',
                                                                                           'description': '搜索关键词，即要查询的地点名称或关键字。'},
                                                                                 'region': {'type': 'string',
                                                                                            'description': '搜索的区域，默认为空字符串，表示不限制区域。',
                                                                                            'default': ''},
                                                                                 'limit': {'type': 'boolean',
                                                                                           'description': '是否限制在指定区域内搜索，默认为True。',
                                                                                           'default': True}},
                                                                  'required': ['query']}}},
                 {'type': 'function', 'function': {
                     'name': 'get_amap_location', 'description': '通过高德地图API获取指定地址的经纬度坐标。',
                     'parameters': {'type': 'object', 'properties': {
                         'address': {'type': 'string', 'description': '要查询的具体地址信息。', 'default': ''},
                         'city': {'type': 'string', 'description': '城市名称，用于辅助提高地址解析的准确性。',
                                  'default': ''}},
                                    'required': ['address']}}},
                 {'type': 'function', 'function': {'name': 'baidu_translate',
                                                   'description': '百度翻译 API',
                                                   'parameters': {'type': 'object',
                                                                  'properties': {
                                                                      'text': {
                                                                          'type': 'string',
                                                                          'description': '需要翻译的文本内容。',
                                                                          'default': ''},
                                                                      'from_lang': {
                                                                          'type': 'string',
                                                                          'description': '源语言代码，默认为中文（zh）。支持的语言代码包括：zh（中文）、en（英文）、ja（日文）、ko（韩文）、fr（法文）、ar（阿拉伯文）、es（西班牙文）、zh-TW（繁体中文）、vi（越南文）等。',
                                                                          'default': 'zh'},
                                                                      'to_lang': {
                                                                          'type': 'string',
                                                                          'description': "目标语言代码，默认为英文（en）。支持的语言代码包括：zh（中文）、en（英文）、ja（日文）、ko（韩文）、fr（法文）、ar（阿拉伯文）、es（西班牙文）、zh-TW（繁体中文）、vi（越南文）等。如果设置为 'auto'，则自动检测源语言并翻译成中文。",
                                                                          'default': 'en'},
                                                                      'trans_type': {
                                                                          'type': 'string',
                                                                          'description': "翻译类型，默认为 'texttrans'。",
                                                                          'default': 'texttrans'}},
                                                                  'required': [
                                                                      'text']}}},
                 {'type': 'function',
                  'function': {'name': 'tencent_translate',
                               'description': '使用腾讯云翻译API将文本从源语言翻译为目标语言。',
                               'parameters': {'type': 'object', 'properties': {
                                   'text': {'type': 'string', 'description': '需要翻译的文本内容。', 'default': ''},
                                   'source': {'type': 'string', 'description': "源语言代码，例如 'en' 表示英语。",
                                              'default': ''},
                                   'target': {'type': 'string', 'description': "目标语言代码，例如 'zh' 表示中文。",
                                              'default': ''}},
                                              'required': ['text', 'source', 'target']}}},
                 {'type': 'function', 'function': {
                     'name': 'xunfei_translate', 'description': '使用讯飞翻译API将文本从源语言翻译成目标语言。',
                     'parameters': {'type': 'object',
                                    'properties': {
                                        'text': {'type': 'string', 'description': '需要翻译的文本内容。', 'default': ''},
                                        'source': {'type': 'string', 'description': "源语言代码，默认为英文（'en'）。",
                                                   'default': 'en'},
                                        'target': {'type': 'string', 'description': "目标语言代码，默认为中文（'cn'）。",
                                                   'default': 'cn'}}, 'required': ['text']}}}, {'type': 'function',
                                                                                                'function': {
                                                                                                    'name': 'search_amap_location',
                                                                                                    'description': '异步函数，用于通过高德地图API搜索地理位置信息。',
                                                                                                    'parameters': {
                                                                                                        'type': 'object',
                                                                                                        'properties': {
                                                                                                            'query': {
                                                                                                                'type': 'string',
                                                                                                                'description': '搜索关键词，必填项。'},
                                                                                                            'region': {
                                                                                                                'type': 'string',
                                                                                                                'description': '搜索区域，默认为空字符串。',
                                                                                                                'default': ''},
                                                                                                            'limit': {
                                                                                                                'type': 'boolean',
                                                                                                                'description': '是否限制在指定区域内搜索，默认为True。',
                                                                                                                'default': True}},
                                                                                                        'required': [
                                                                                                            'query']}}},
                 {'type': 'function', 'function': {'name': 'web_search_async',
                                                   'description': '异步执行网页搜索，使用给定的文本和API密钥，并返回搜索结果。',
                                                   'parameters': {'type': 'object', 'properties': {
                                                       'text': {'type': 'string', 'description': '要搜索的文本内容。',
                                                                'default': ''},
                                                       'api_key': {'type': 'string',
                                                                   'description': '用于授权的API密钥。',
                                                                   'default': 'Config.GLM_Service_Key'},
                                                       'kwargs': {'type': 'object',
                                                                  'description': '额外的关键字参数，用于扩展请求数据。',
                                                                  'additionalProperties': True, 'default': {}}},
                                                                  'required': ['text']}}},
                 {'type': 'function', 'function': {
                     'name': 'web_search_tavily',
                     'description': '使用 Tavily API 进行网页搜索，支持指定主题、时间范围、搜索深度和天数等参数。',
                     'parameters': {'type': 'object',
                                    'properties': {
                                        'text': {'type': 'string', 'description': '要搜索的文本或查询字符串。'},
                                        'topic': {'type': 'string',
                                                  'description': "搜索的主题，可以是 'general' 或 'news'。",
                                                  'default': 'general'}, 'time_range': {'type': 'string',
                                                                                        'description': "搜索的时间范围，可以是 'day', 'week', 'month', 'year', 'd', 'w', 'm', 'y'。",
                                                                                        'default': 'month'},
                                        'search_depth': {'type': 'string',
                                                         'description': "搜索的深度，可以是 'basic' 或 'advanced'。",
                                                         'default': 'basic'},
                                        'days': {'type': 'integer',
                                                 'description': "如果主题是 'news'，则指定搜索的天数。",
                                                 'default': 7},
                                        'api_key': {'type': 'string', 'description': 'Tavily API 的密钥。',
                                                    'default': 'Config.TAVILY_Api_Key'}, 'kwargs': {'type': 'object',
                                                                                                    'description': '其他可选参数，以关键字参数的形式传递。'}},
                                    'required': ['text']}}},
                 {'type': 'function', 'function': {'name': 'web_extract_tavily',
                                                   'description': '通过 Tavily API 提取给定 URL 的信息。',
                                                   'parameters': {'type': 'object',
                                                                  'properties': {
                                                                      'urls': {
                                                                          'type': 'array',
                                                                          'items': {
                                                                              'type': 'string'},
                                                                          'description': '要提取信息的 URL 列表。'},
                                                                      'api_key': {
                                                                          'type': 'string',
                                                                          'description': 'Tavily API 的访问密钥。',
                                                                          'default': 'Config.TAVILY_Api_Key'}},
                                                                  'required': [
                                                                      'urls']}}},
                 {'type': 'function', 'function': {'name': 'wikipedia_search',
                                                   'description': '通过维基百科API搜索查询内容，并返回第一个搜索结果的摘要信息。',
                                                   'parameters': {'type': 'object', 'properties': {
                                                       'query': {'type': 'string',
                                                                 'description': '要搜索的查询字符串。'}},
                                                                  'required': ['query']}}},
                 {'type': 'function', 'function': {
                     'name': 'baidu_translate', 'description': '百度翻译 API',
                     'parameters': {'type': 'object', 'properties': {
                         'text': {'type': 'string', 'description': '需要翻译的文本内容。', 'default': ''},
                         'from_lang': {'type': 'string',
                                       'description': '源语言代码，默认为中文（zh）。支持的语言代码包括：zh（中文）、en（英文）、ja（日文）、ko（韩文）、fr（法文）、ar（阿拉伯文）、es（西班牙文）、zh-TW（繁体中文）、vi（越南文）等。',
                                       'default': 'zh'}, 'to_lang': {'type': 'string',
                                                                     'description': "目标语言代码，默认为英文（en）。支持的语言代码包括：zh（中文）、en（英文）、ja（日文）、ko（韩文）、fr（法文）、ar（阿拉伯文）、es（西班牙文）、zh-TW（繁体中文）、vi（越南文）等。如果设置为 'auto'，则自动检测源语言并翻译成中文。",
                                                                     'default': 'en'},
                         'trans_type': {'type': 'string', 'description': "翻译类型，默认为 'texttrans'。",
                                        'default': 'texttrans'}},
                                    'required': ['text']}}},
                 {'type': 'function', 'function': {'name': 'call_http_request',
                                                   'description': '这是一个异步函数，用于发送HTTP GET请求并返回响应的JSON数据。',
                                                   'parameters': {'type': 'object', 'properties': {
                                                       'url': {'type': 'string', 'description': '请求的目标URL地址。'},
                                                       'headers': {'type': 'object',
                                                                   'description': 'HTTP请求头信息，以字典形式传递。',
                                                                   'default': None},
                                                       'time_out': {'type': 'number',
                                                                    'description': '请求的超时时间，单位为秒。',
                                                                    'default': 100.0}, 'kwargs': {'type': 'object',
                                                                                                  'description': '其他关键字参数，用于传递给httpx.get方法。'}},
                                                                  'required': ['url']}}}]

'''
功能型 (function): 核心逻辑的实现。
工具型 (tool): 通过现成工具提供结果。
API型 (api): 封装外部接口调用。
服务型 (service): 长时间运行或后台任务。
插件型 (plugin): 为现有系统提供扩展。
数据型 (data): 处理和生成数据的任务。
查询型 (query): 面向数据检索的操作。
'''
