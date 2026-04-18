"""Tool para consultar el tiempo usando Open-Meteo (API gratuita, sin key)."""
import requests
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Obtiene el tiempo actual y pronóstico de 3 días de cualquier ciudad."""
    try:
        # 1. Geocoding: ciudad -> lat/lon
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "es"},
            timeout=10,
        )
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

        if not geo_data.get("results"):
            return f"No se encontró la ciudad '{city}'."

        location = geo_data["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]
        name = location["name"]
        country = location.get("country", "")

        # 2. Weather
        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto",
                "forecast_days": 3,
            },
            timeout=10,
        )
        weather_resp.raise_for_status()
        data = weather_resp.json()

        current = data.get("current", {})
        daily = data.get("daily", {})

        weather_codes = {
            0: "Despejado", 1: "Mayormente despejado", 2: "Parcialmente nublado",
            3: "Nublado", 45: "Niebla", 48: "Niebla con escarcha",
            51: "Llovizna ligera", 53: "Llovizna", 55: "Llovizna intensa",
            61: "Lluvia ligera", 63: "Lluvia", 65: "Lluvia intensa",
            71: "Nieve ligera", 73: "Nieve", 75: "Nieve intensa",
            80: "Chubascos ligeros", 81: "Chubascos", 82: "Chubascos intensos",
            95: "Tormenta", 96: "Tormenta con granizo ligero", 99: "Tormenta con granizo",
        }
        condition = weather_codes.get(current.get("weather_code"), "Desconocido")

        lines = [
            f"Tiempo en {name}, {country}:",
            f"- Temperatura actual: {current.get('temperature_2m')}°C",
            f"- Condición: {condition}",
            f"- Humedad: {current.get('relative_humidity_2m')}%",
            f"- Viento: {current.get('wind_speed_10m')} km/h",
            "\nPronóstico próximos 3 días:",
        ]

        dates = daily.get("time", [])
        tmax = daily.get("temperature_2m_max", [])
        tmin = daily.get("temperature_2m_min", [])
        prec = daily.get("precipitation_sum", [])

        for i in range(min(3, len(dates))):
            lines.append(
                f"- {dates[i]}: {tmin[i]}°C / {tmax[i]}°C, "
                f"precipitación {prec[i]}mm"
            )

        return "\n".join(lines)
    except requests.RequestException as e:
        return f"Error de red consultando el tiempo: {e}"
    except Exception as e:
        return f"Error obteniendo el tiempo: {e}"
