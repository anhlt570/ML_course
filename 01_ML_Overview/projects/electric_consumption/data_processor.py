import numpy
import pandas
import datetime_utils


def BinarySearch(hour, hour_list):
    if len(hour_list) == 0:
        return -1
    if len(hour_list) == 1:
        return -1 if hour != hour_list[0] else 0

    midle_pos = int(len(hour_list)/2)
    if hour_list[midle_pos] == hour:
        return midle_pos
    elif hour_list[midle_pos] > hour:
        return BinarySearch(hour, hour_list[:midle_pos])
    else:
        return midle_pos + BinarySearch(hour, hour_list[midle_pos:])


def GetData():
    electric_data = pandas.read_csv('data/nyc_demand.csv').values
    weather_data = pandas.read_csv('data/nyc_weather.csv').values

    electric_data_hours = numpy.array([
        datetime_utils.GetHours(electric_data[i][0]) for i in range(0, electric_data.shape[0])
    ])
    weather_data_hours = numpy.array([
        datetime_utils.GetHours(weather_data[i][0]) for i in range(0, weather_data.shape[0])
    ])
    X = []
    Y = []
    for i in range(0, electric_data_hours.shape[0]):
        hour_data =electric_data_hours[i]
        pos = BinarySearch( hour_data, weather_data_hours)
        if pos != -1:
            x=[]
            datetime = datetime_utils.GetTime(electric_data[i][0])
            month = datetime_utils.month_to_code(datetime.month)
            day_of_week = datetime_utils.GetDayOfWeek(datetime)
            day_of_week =  datetime_utils.day_to_code(day_of_week)
            hour = datetime_utils.hour_to_code(datetime.hour)
            x+=  month + day_of_week + hour
            temperature = weather_data[pos][1]
            x.append(temperature)
            X.append(x)

            electric_consumption = electric_data[i][1]
            Y.append(electric_consumption)
    return (numpy.array(X), numpy.array(Y))
