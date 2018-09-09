import datetime


def GetTime(time_str):
    fomat = '%Y-%m-%d %H:%M:%S'
    return datetime.datetime.strptime(time_str, fomat)


def CalculateHourInterval(start_str, end_str):
    time_delta = GetTime(end_str) - GetTime(start_str)
    return divmod(time_delta.days * 86400 + time_delta.seconds, 3600)[0]


def GetHours(date_str):
    begin_date_str = '1970-01-01 00:00:00'
    return CalculateHourInterval(begin_date_str, date_str)


def GetDayOfWeek(datetime):
    return datetime.today().weekday()


def hour_to_code(hour):
    code = []
    for i in range(0, 23):
        code.append(1.0 if i == hour else 0.0)
    return code


def day_to_code(day_of_week):
    code = []
    for i in range(0, 6):
        code.append(1.0 if i == day_of_week else 0.0)
    return code


def month_to_code(month):
    code = []
    for i in range(0, 11):
        code.append(1.0 if i == month else 0.0)
    return code
