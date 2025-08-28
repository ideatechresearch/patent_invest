from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
import time


def get_clock(t, speed=10):
    return "🕐🕑🕒🕓🕔🕕🕖🕗🕘🕙🕚🕛"[int(t * speed) % 12]


def timeframe_sec(frequency='1m'):
    frequency_dict = {'default': 0, 'tick': 0.9, '5s': 5, '1m': 60, '5m': 300, '15m': 900, '20m': 1200, '30m': 1800,
                      '60m': 3600, '1h': 3600, '4h': 14400, '1d': 86400,
                      'week': 604800, 'month': 2628000, 'quarter': 788400, 'halfyear': 15768000, 'year': 31536000}
    return frequency_dict.get(frequency, 0)


def float_time(timetuple, datetype=0):
    if timetuple is None:
        return 0.0
    date_s = timetuple.year * 10000000000 + timetuple.month * 100000000 + timetuple.day * 1000000
    if datetype == 1:
        return date_s
    time_s = float(timetuple.hour * 10000 + timetuple.minute * 100 + timetuple.second)
    if datetype == 2:
        return time_s
    return date_s + time_s


def seconds_time(t_to, t_from=0):
    return time.mktime(t_to.timetuple()) if t_from == 0 else (t_to - t_from).total_seconds()


def getime_period(prevtime, now, pernow, perstep=300):
    '''
    将时间对齐到指定步长周期。 返回对齐后的时间
    calc time in period by prev and step
    '''
    if prevtime is None:  # 初始化itime(0)
        prevtime = pernow

    shift = int((now - prevtime).total_seconds() // perstep)
    thistime = prevtime + timedelta(seconds=shift * perstep)

    if pernow is not None:
        if pernow > thistime:
            return pernow
        excess = (thistime - pernow).total_seconds() % perstep
        if excess:
            thistime -= timedelta(seconds=excess)
    return thistime


def parse_time(val):
    # 解析时间字符串，并转换为 datetime 对象。如果解析失败，返回一个默认值 datetime.min（0001-01-01 00:00:00）。
    if isinstance(val, datetime):
        return val
    try:
        return datetime.strptime(val, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.min


def parse_time_format(val):
    # 将 datetime 对象转换为 ISO 8601 格式的字符串
    if isinstance(val, datetime):
        return val.isoformat()  # 转换为ISO 8601格式
    return val


def format_date(date_str: str):
    # 直接使用 strptime 来解析日期并格式化为目标格式：YYYY-MM-DD
    return datetime.strptime(date_str, "%Y/%m/%d %H:%M:%S").date().strftime("%Y-%m-%d")


def unix_timestamp(iso_time_str: str) -> float:
    '''2025-05-21T06:30:59.6050065Z'''
    # 去除多余的微秒部分，限制到 6 位（Python datetime 支持的上限）
    iso_time_str = iso_time_str[:26] + 'Z'
    # 转换为 datetime 对象
    dt = datetime.strptime(iso_time_str, '%Y-%m-%dT%H:%M:%S.%fZ')
    # 设置为 UTC 时区
    dt = dt.replace(tzinfo=timezone.utc)
    # 转换为 Unix 时间戳（float）
    return dt.timestamp()


def format_date_type(date=None):
    """
    :param date:可以是一个日期字符串或 None（如果传入 None，则使用当前日期）。
    :return:返回一个 datetime 对象，如果 date 是有效的字符串，或者当前时间或datetime
    """
    # 如果没有传入日期，使用当前日期
    if not date:
        date = datetime.now()
    elif isinstance(date, str):
        supported_formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            '%Y%m%d',
        ]
        for fmt in supported_formats:
            try:
                date = datetime.strptime(date, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Invalid date format: {date}. Supported formats are {supported_formats}.")

    return date  # isinstance(date, datetime)


def get_date_time(date=None):
    """
    :param date: 指定的日期（默认为当前日期）。"%B %d, %Y“
    :return: 日期，时间
    """
    date = format_date_type(date)
    return date.strftime('%Y%m%d'), date.strftime('%H:%M:%S.%f')


def get_times_shift(days_shift: int = 0, hours_shift: int = 0):
    """
    :param days_shift: 偏移的天数，>0 表示未来，<0 表示过去，0 表示当前日期。
    :param hours_shift: 偏移的小时数，>0 表示未来，<0 表示过去，0 表示当前时间。
    :return: 格式化后的时间，格式为 'YYYY-MM-DD HH:MM:SS'。
    """
    current_datetime = datetime.now()
    adjusted_time = current_datetime + timedelta(days=days_shift, hours=hours_shift)
    return adjusted_time.strftime('%Y-%m-%d %H:%M:%S')


def get_day_range(date=None, shift: int = 0, count: int = 1):
    date = format_date_type(date)
    start_date = date - timedelta(days=shift)
    end_date = start_date + timedelta(days=count)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def get_week_range(date=None, shift: int = 0, count: int = 1):
    """
    获取指定日期所在周的开始和结束日期。
    支持通过 shift 参数偏移周。

    :param date: 指定的日期（默认为当前日期）。
    :param shift: 偏移周数，>0 表示未来的周，<0 表示过去的周，0 表示当前周。
    :param count: 控制返回的周数范围，默认为 1，表示返回一个周的日期范围。
    :return: 返回指定周的开始和结束日期，格式为 ('YYYY-MM-DD', 'YYYY-MM-DD')
    """
    date = format_date_type(date)
    # 根据 shift 参数调整日期
    date = date + relativedelta(weeks=shift)
    # 获取今天是周几 (0 是周一, 6 是周日)
    weekday = date.weekday()
    # 计算周一的日期 (开始日期)
    start_of_week = date - timedelta(days=weekday)
    # 计算周日的日期 (结束日期)
    # end_of_week = start_of_week + timedelta(days=6)

    end_date = start_of_week + timedelta(weeks=count) - timedelta(days=1)  # start_of_week + timedelta(days=6)

    return start_of_week.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def get_month_range(date=None, shift: int = 0, count: int = 1):
    """
    获取指定日期所在月的开始和结束日期。
    支持通过 shift 参数偏移月数，和通过 count 控制返回的月份范围。

    :param date: 指定的日期（默认为当前日期）。
    :param shift: 偏移的月数，>0 表示未来的月，<0 表示过去的月，0 表示当前月。
    :param count: 控制返回的月份范围，默认为 1，表示返回一个月的开始和结束日期。
    :return: 返回指定月份的开始和结束日期，格式为 ('YYYY-MM-DD', 'YYYY-MM-DD')
      """
    date = format_date_type(date)
    # 根据 shift 参数调整日期
    start_date = (date + relativedelta(months=shift)).replace(day=1)
    # 计算下个月的第一天，然后减去一天
    end_date = (start_date + relativedelta(months=count)).replace(day=1) - timedelta(days=1)  # + timedelta(days=32)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')  # '%Y-%m-01'


def get_quarter_range(date=None, shift: int = 0, count: int = 1):
    """
    获取指定日期所在季度的开始和结束日期。
    支持通过 shift 参数偏移季度数，和通过 count 控制返回的季度范围。

    :param date: 指定的日期（默认为当前日期）。
    :param shift: 偏移的季度数，>0 表示未来的季度，<0 表示过去的季度，0 表示当前季度。
    :param count: 控制返回的季度范围，默认为 1，表示返回一个季度的开始和结束日期。
    :return: 返回指定季度的开始和结束日期，格式为 ('YYYY-MM-DD', 'YYYY-MM-DD')
    """
    date = format_date_type(date)

    # 确定当前日期所在季度的起始月份
    start_month = 3 * ((date.month - 1) // 3) + 1
    start_date = (date.replace(month=start_month, day=1)
                  + relativedelta(months=3 * shift))

    # 计算季度结束日期
    end_date = (start_date + relativedelta(months=3 * count)).replace(day=1) - timedelta(days=1)

    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def get_quarter_month_range(date=None, shift: int = 0, count: int = 1):
    """
    获取指定日期所在季度的月份范围。
    支持通过 shift 参数偏移季度数，和通过 count 控制返回的季度范围。

    :param date: 指定的日期（默认为当前日期）。
    :param shift: 偏移季度数，>0 表示未来的季度，<0 表示过去的季度，0 表示当前季度。
    :param count: 控制返回的季度范围，默认为 1，表示返回一个季度的开始和结束日期。
    :return: 返回季度的开始月和结束月以及起始年份，格式为 ('YYYY','MM', 'MM')
    """
    date = format_date_type(date)

    current_year = date.year

    # 计算当前季度的起始月份
    quarter_start = (date.month - 1) // 3 * 3 + 1

    # 根据 shift 偏移季度
    quarter_start += shift * 3

    # 处理跨年情况：如果起始月份超出了12月，需要调整年份
    if quarter_start > 12:
        quarter_start -= 12
        current_year += 1
    elif quarter_start < 1:
        quarter_start += 12
        current_year -= 1

    # 计算季度的结束月份
    quarter_end = quarter_start + 3 * count - 1

    # 处理结束月份跨年情况：如果结束月份超过12月，需要调整年份
    if quarter_end > 12:
        quarter_end -= 12

    return current_year, quarter_start, quarter_end


def get_year_range(date=None, shift: int = 0, count: int = 1):
    """
    获取指定日期所在年的开始和结束日期。
    支持通过 shift 参数偏移年数，和通过 count 控制返回的年度范围。

    :param date: 指定的日期（默认为当前日期）。
    :param shift: 偏移的年数，>0 表示未来的年，<0 表示过去的年，0 表示当前年。
    :param count: 控制返回的年度范围，默认为 1，表示返回一年的开始和结束日期。
    :return: 返回指定年的开始和结束日期，格式为 ('YYYY-MM-DD', 'YYYY-MM-DD')
    """
    date = format_date_type(date)

    # 计算年份的开始日期
    start_date = date.replace(month=1, day=1) + relativedelta(years=shift)

    # 计算年份的结束日期
    end_date = (start_date + relativedelta(years=count)).replace(day=1, month=1) - timedelta(days=1)

    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def get_half_year_range(date=None, shift: int = 0, count: int = 1):
    """
    获取指定日期所在的半年（前半年或后半年）范围。
    支持通过 shift 参数偏移半年数，和通过 count 控制返回的半年数范围。

    :param date: 指定的日期（默认为当前日期）。
    :param shift: 半年偏移量，0 表示当前半年，-1 表示前一半年，1 表示下一半年。
    :param count: 返回的半年范围，默认为 1，表示返回一个半年的开始和结束日期。
    :return: 返回指定半年的开始和结束日期，格式为 ('YYYY-MM-DD', 'YYYY-MM-DD')
    """
    date = format_date_type(date)

    # 判断当前是前半年还是后半年
    if date.month <= 6:
        start_date = date.replace(month=1, day=1)
    else:
        start_date = date.replace(month=7, day=1)

    # 调整日期到指定的半年
    start_date += relativedelta(months=6 * shift)
    # 计算半年结束日期
    end_date = (start_date + relativedelta(months=6 * count)).replace(day=1) - timedelta(days=1)

    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def date_range_calculator(period_type: str, date=None, shift: int = 0, count: int = 1) -> dict:
    """
    计算基于参考日期的时间范围。

    :param period_type: 时间周期类型，'days'、'weeks'、'months' 等
    :param date: 基准日期，格式为 'YYYY-MM-DD'
    :param shift: 半年偏移量，0 表示当前半年，-1 表示前一半年，1 表示下一半年。
    :param count: 时间周期数量，表示从参考日期向前或向后的时长
    :return: 返回计算出的日期范围，包含 'start_date' 和 'end_date'
    """
    period_map = {'days': get_day_range,
                  'weeks': get_week_range,
                  'month': get_month_range,
                  'quarters': get_quarter_range,
                  'half_year': get_half_year_range,
                  'year': get_year_range,
                  }

    handler = period_map.get(period_type)
    if handler:
        start_date, end_date = handler(date, shift, count)
    else:
        raise ValueError(f"不支持的时间单位: {period_type}")

    # 返回结果字典，包含开始和结束日期
    return {'start_date': start_date, 'end_date': end_date}
