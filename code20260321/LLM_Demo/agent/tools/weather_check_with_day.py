# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/21 15:28
Create User : 19410
Desc : xxx
"""

from pydantic import BaseModel, Field
import requests


def weather(location: str, days: int, api_key: str):
    url = f"https://api.seniverse.com/v3/weather/daily.json?key={api_key}&location={location}&language=zh-Hans&unit=c&start=0&days={days}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        daily = data["results"][0]['daily']
        weather = []
        for day in daily:
            weather.append({
                'date': day['date'],
                "low temperature": day["low"],
                "high temperature": day["high"],
                "description": day["text_day"],
            })
        return weather
    else:
        raise Exception(
            f"Failed to retrieve weather: {response.status_code}")


def weathercheck_with_days(location, days):
    print(f"入参:{location} - {days} - {type(location)} - {type(days)}")
    return weather(location, days, "SqWCDI5TuUyD4Nbby")


class WeatherInputWithDays(BaseModel):
    location: str = Field(description="City name,include city and county")
    days: int = Field(description="需要查询的未来几天的天气 所对应的天数")
