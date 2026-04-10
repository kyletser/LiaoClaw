"""
简单的天气 MCP 服务器
使用 Open-Meteo API (免费无需 API Key)
"""
from fastmcp import FastMCP
import httpx
import json

mcp = FastMCP("Weather Server")

# Open-Meteo 是免费的天气 API，不需要 API Key
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


@mcp.tool()
async def get_weather(city: str, country: str = "") -> str:
    """
    获取指定城市的天气信息
    
    Args:
        city: 城市名称 (如 "Beijing", "Shanghai")
        country: 国家代码 (可选，如 "CN", "US")
    
    Returns:
        天气信息字符串
    """
    async with httpx.AsyncClient() as client:
        # 1. First get city coordinates - try both English and Chinese
        search_params = {"name": city, "count": 5, "language": "en", "format": "json"}
        
        # Try English first
        resp = await client.get(GEOCODING_URL, params=search_params)
        data = resp.json()
        
        # If not found, try Chinese
        if not data.get("results"):
            search_params = {"name": city, "count": 5, "language": "zh", "format": "json"}
            resp = await client.get(GEOCODING_URL, params=search_params)
            data = resp.json()
        
        if not data.get("results"):
            # Try without language param
            search_params = {"name": city, "count": 5, "format": "json"}
            resp = await client.get(GEOCODING_URL, params=search_params)
            data = resp.json()
        
        if not data.get("results"):
            return f"找不到城市: {city}，请尝试使用英文名称如 Wuhan"
        
        location = data["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]
        name = location["name"]
        country_name = location.get("country", "")
        
        # 2. 获取天气数据
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,weather_code",
            "timezone": "auto",
            "forecast_days": 3,
        }
        
        resp = await client.get(WEATHER_URL, params=weather_params)
        weather_data = resp.json()
        
        current = weather_data.get("current", {})
        daily = weather_data.get("daily", {})
        
        # 格式化天气描述
        weather_code = current.get("weather_code", 0)
        weather_desc = _weather_code_description(weather_code)
        
        result = f"""🌤️ {name}, {country_name}
===================
当前天气:
  温度: {current.get('temperature_2m', 'N/A')}°C
  湿度: {current.get('relative_humidity_2m', 'N/A')}%
  风速: {current.get('wind_speed_10m', 'N/A')} km/h
  状况: {weather_desc}

未来几天预报:
"""
        for i, date in enumerate(daily.get("time", [])[:3]):
            max_temp = daily.get("temperature_2m_max", [])[i]
            min_temp = daily.get("temperature_2m_min", [])[i]
            code = daily.get("weather_code", [])[i]
            desc = _weather_code_description(code)
            result += f"  {date}: {min_temp}°C ~ {max_temp}°C {desc}\n"
        
        return result


@mcp.tool()
async def get_forecast(city: str, days: int = 7) -> str:
    """
    获取指定城市的多日天气预报
    
    Args:
        city: 城市名称
        days: 预报天数 (1-16，默认 7)
    
    Returns:
        天气预报字符串
    """
    days = min(max(days, 1), 16)  # 限制在 1-16 天
    
    async with httpx.AsyncClient() as client:
        # 获取城市坐标
        resp = await client.get(GEOCODING_URL, params={"name": city, "count": 1})
        data = resp.json()
        
        if not data.get("results"):
            return f"找不到城市: {city}"
        
        location = data["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]
        name = location["name"]
        
        # 获取天气
        resp = await client.get(WEATHER_URL, params={
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": days,
        })
        weather_data = resp.json()
        
        daily = weather_data.get("daily", {})
        result = f"📅 {name} {days} 天天气预报\n========================\n"
        
        for i, date in enumerate(daily.get("time", [])):
            max_temp = daily.get("temperature_2m_max", [])[i]
            min_temp = daily.get("temperature_2m_min", [])[i]
            code = daily.get("weather_code", [])[i]
            precip = daily.get("precipitation_probability_max", [])[i]
            desc = _weather_code_description(code)
            
            result += f"{date}: {min_temp}°C ~ {max_temp}°C | {desc} | 降水概率: {precip}%\n"
        
        return result


def _weather_code_description(code: int) -> str:
    """将天气代码转换为描述"""
    codes = {
        0: "☀️ 晴",
        1: "🌤️ 晴间多云",
        2: "⛅ 多云",
        3: "☁️ 阴",
        45: "🌫️ 雾",
        48: "🌫️ 雾凇",
        51: "🌧️ 小雨",
        53: "🌧️ 中雨",
        55: "🌧️ 大雨",
        61: "🌧️ 雨",
        63: "🌧️ 中雨",
        65: "🌧️ 大雨",
        71: "🌨️ 小雪",
        73: "🌨️ 中雪",
        75: "🌨️ 大雪",
        77: "🌨️ 雪粒",
        80: "🌧️ 阵雨",
        81: "🌧️ 小阵雨",
        82: "🌧️ 大阵雨",
        85: "🌨️ 小阵雪",
        86: "🌨️ 大阵雪",
        95: "⛈️ 雷暴",
        96: "⛈️ 雷暴 + 冰雹",
        99: "⛈️ 雷暴 + 大冰雹",
    }
    return codes.get(code, f"❓ 代码 {code}")


if __name__ == "__main__":
    # 使用 stdio 传输模式运行
    mcp.run(transport="stdio")