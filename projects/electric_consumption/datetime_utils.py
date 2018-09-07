import datetime

s = '2012-03-31 01:00:00'
date = datetime.datetime.strptime(s,"%Y-%m-%d %H:%M:%S")
print(date.year)
print(date.month)
print(date.day)
print(date.hour)
print(date.minute)
print(date.second)




