AI_Tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "当你想知道现在的时间时非常有用。",
            "parameters": {}  # 此处是函数参数相关描述, 因为获取当前时间无需输入参数，因此parameters为空字典
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_week_range",
            "description": "获取指定日期所在周的开始和结束日期，支持偏移周数和返回多个周的范围。",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "format": "date",
                        "description": "指定的日期，格式为 'YYYY-MM-DD'，默认为当前日期。",
                    },
                    "shift": {
                        "type": "integer",
                        "description": "偏移的周数，>0 表示未来的周，<0 表示过去的周，0 表示当前周。",
                    },
                    "count": {
                        "type": "integer",
                        "description": "控制返回的周数范围，默认为 1，表示返回一个周的范围。",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_first_day_of_month",
            "description": "获取指定日期所在月的第一天。",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "format": "date",
                        "description": "指定的日期，格式为 'YYYY-MM-DD'，默认为当前日期。",
                    },
                    "shift": {
                        "type": "integer",
                        "description": "偏移的月数，>0 表示未来的月，<0 表示过去的月，0 表示当前月。",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_month_range",
            "description": "获取指定日期所在月的开始和结束日期，支持偏移月数和返回多个月份范围。",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "format": "date",
                        "description": "指定的日期，格式为 'YYYY-MM-DD'，默认为当前日期。",
                    },
                    "shift": {
                        "type": "integer",
                        "description": "偏移的月数，>0 表示未来的月，<0 表示过去的月，0 表示当前月。",
                    },
                    "count": {
                        "type": "integer",
                        "description": "控制返回的月份范围，默认为 1，表示返回一个月的范围。",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_quarter_range",
            "description": "获取指定日期所在季度的开始和结束日期，支持偏移季度数和返回多个季度范围。",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "format": "date",
                        "description": "指定的日期，格式为 'YYYY-MM-DD'，默认为当前日期。",
                    },
                    "shift": {
                        "type": "integer",
                        "description": "偏移的季度数，>0 表示未来的季度，<0 表示过去的季度，0 表示当前季度。",
                    },
                    "count": {
                        "type": "integer",
                        "description": "控制返回的季度范围，默认为 1，表示返回一个季度的范围。",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_year_range",
            "description": "获取指定日期所在年的开始和结束日期，支持偏移年数和返回多个年度范围。",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "format": "date",
                        "description": "指定的日期，格式为 'YYYY-MM-DD'，默认为当前日期。",
                    },
                    "shift": {
                        "type": "integer",
                        "description": "偏移的年数，>0 表示未来的年，<0 表示过去的年，0 表示当前年。",
                    },
                    "count": {
                        "type": "integer",
                        "description": "控制返回的年度范围，默认为 1，表示返回一年的范围。",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_half_year_range",
            "description": "获取指定日期所在半年的开始和结束日期，支持偏移半年和返回多个半年的范围。",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "format": "date",
                        "description": "指定的日期，格式为 'YYYY-MM-DD'，默认为当前日期。",
                    },
                    "shift": {
                        "type": "integer",
                        "description": "偏移的半年数，>0 表示未来的半年，<0 表示过去的半年，0 表示当前半年。",
                    },
                    "count": {
                        "type": "integer",
                        "description": "控制返回的半年范围，默认为 1，表示返回一个半年的范围。",
                    },
                },
                "required": [],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Get the current weather for a given city.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'city': {
                        'type': 'string',
                        'description': 'The name of the city to query weather for.',
                    },
                },
                'required': ['city'],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "通过提供的查询文本执行网络搜索。",
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
    }
]
