import requests
from typing import Dict

import os
from dotenv import load_dotenv

import pandas as pd
import random

class APIManager:
    BASE_URL = "https://servicebus.ren.pt/datahubapi"

    def __init__(self, lang = "en-US"):
        self.lang = lang
        self.session = requests.Session()

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.datafiles_dir = os.path.join(self.base_dir, "datafiles")
        
        self.price_data_filename = "market_prices.csv"
        self.solar_data_filename = "solar_production.csv"
        self.wind_data_filename = "wind_production.csv"
        self.consumption_data_filename = "consumption.csv"

    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        params["culture"] = self.lang
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Erro ao consultar API: {e}")
        return None    
    
    def get_daily_price(self, date: str):
        data = self._make_request(
            "electricity/ElectricityMarketPricesDaily",
            {"date": date}
        )
        return data
    
    def get_daily_production_breakdown(self, date: str):
        data = self._make_request(
            "electricity/ElectricityProductionBreakdownDaily",
            {"date": date}
        )
        return data
    
    def generate_data(self, date: str):
        price_data = self.get_daily_price(date)
        production_data = self.get_daily_production_breakdown(date)
        
        folder_path = os.path.join(self.datafiles_dir, date)
        os.makedirs(folder_path, exist_ok=True)

        self.gen_market_data(price_data, folder_path)
        self.gen_production_data(production_data, folder_path)
        self.gen_consumption_data(folder_path)


    def gen_production_data(self, data: Dict, folder_path: str):
        time_unit = data.get("xAxis", {}).get("title", {}).get("text")
        time_series = data.get("xAxis", {}).get("categories")
        
        production_unit = data.get("yAxis", {}).get("title", {}).get("text")
        production_series_list = data.get("series", [])
        solar_series = []
        wind_series = []

        skip_flag = False
        for series in production_series_list:
            source = series.get("name")

            if source == "Solar":
                solar_series = series.get("data", [])
                if skip_flag:
                    break
                skip_flag = True

            if source == "Wind":
                wind_series = series.get("data", [])
                if skip_flag:
                    break
                skip_flag = True
        
        solar_series_kw = [value * 1_000 / 500_000 for value in solar_series]  
        wind_series_kw = [value * 1_000 / 500_000 for value in wind_series] 
        
        df_solar = pd.DataFrame({
            f"Time ({time_unit})": time_series,
            f"Production (kW)": solar_series_kw
        })

        df_wind = pd.DataFrame({
            f"Time ({time_unit})": time_series,
            f"Production (kW)": wind_series_kw
        })

        solar_csv_path = os.path.join(folder_path, self.solar_data_filename)
        wind_csv_path = os.path.join(folder_path, self.wind_data_filename)

        df_solar.to_csv(solar_csv_path, index=False)
        df_wind.to_csv(wind_csv_path, index=False)
        
        print(f"Solar production saved to {solar_csv_path}")
        print(f"Wind production saved to {wind_csv_path}")

        return df_solar, df_wind


    def gen_market_data(self, data: Dict, folder_path: str):
        time_unit = data.get("xAxis", {}).get("title", {}).get("text")
        time_series = data.get("xAxis", {}).get("categories")
        
        production_unit = data.get("yAxis", {}).get("title", {}).get("text")
        price_series_list = data.get("series", [])
        price_series = []

        for series in price_series_list:
            source = series.get("name")
            if source == "PT":
                price_series = series.get("data", [])
                break

        formatted_times = []
        filtered_prices = []
        
        hour_24_value = None
        for i, time_val in enumerate(time_series):
            try:
                hour = int(time_val)
                if hour == 24:
                    hour_24_value = price_series[i]
                    break
            except (ValueError, TypeError):
                continue
        
        if hour_24_value is not None:
            formatted_times.append("00:00")
            filtered_prices.append(hour_24_value)
        
        for i, time_val in enumerate(time_series):
            try:
                hour = int(time_val) 
                if hour == 24: 
                    continue
                if hour == 0: 
                    continue
                formatted_times.append(f"{hour:02d}:00")
                filtered_prices.append(price_series[i])
            except (ValueError, TypeError):
                formatted_times.append(str(time_val))
                filtered_prices.append(price_series[i])

        filtered_prices_kwh = [price / 1000 for price in filtered_prices]

        df = pd.DataFrame({
            'Time (Hour)': formatted_times,
            'Price (€/kWh)': filtered_prices_kwh
        })
        
        csv_path = os.path.join(folder_path, self.price_data_filename)
        df.to_csv(csv_path, index=False)
        print(f"Market data saved to {csv_path}")
        
        return df

    def gen_consumption_data(self, folder_path: str):
        times = [f"{h:02d}:{m:02d}" for h in range(24) for m in [0, 15, 30, 45]]
        
        # Consumption pattern (in kW) - typical household
        # Night (00:00-06:00): Low consumption ~0.5-0.8kW
        # Morning (06:00-09:00): Rising consumption ~0.8-2.0kW
        # Midday (09:00-12:00): Moderate ~0.6-1.0kW
        # Afternoon (12:00-18:00): Moderate ~0.7-1.2kW
        # Evening peak (18:00-23:00): High ~1.5-3.0kW
        # Late night (23:00-00:00): Declining ~1.0-0.5kW
        
        consumption = []
        for i, time in enumerate(times):
            hour = i // 4
            
            if 0 <= hour < 6: 
                base = 0.6
                variation = 0.1
            elif 6 <= hour < 9:  
                base = 0.8 + (hour - 6) * 0.4
                variation = 0.2
            elif 9 <= hour < 12: 
                base = 0.8
                variation = 0.15
            elif 12 <= hour < 14: 
                base = 1.2
                variation = 0.2
            elif 14 <= hour < 18: 
                base = 0.9
                variation = 0.15
            elif 18 <= hour < 22: 
                base = 2.0 + (hour - 18) * 0.2
                variation = 0.3
            else:  
                base = 1.5 - (hour - 22) * 0.4
                variation = 0.2
            
            value = base + random.uniform(-variation, variation)
            consumption.append(max(0.3, value))  
        
        df_consumption = pd.DataFrame({
            'Time (Hour)': times,
            'Consumption (kW)': consumption
        })
        
        csv_path = os.path.join(folder_path, self.consumption_data_filename)
        df_consumption.to_csv(csv_path, index=False)
        print(f"Consumption data saved to {csv_path}")
        
        return df_consumption

        
api_manager = APIManager()

if __name__ == "__main__":
    load_dotenv()
    date = os.getenv("DATE", "2025-02-25")

    print("===========TEST API CALLS============")
    print("Daily Prices:")
    print(api_manager.get_daily_price(date))
    print("\nDaily Production Breakdown:")
    print(api_manager.get_daily_production_breakdown(date))

    print("===========GENERATE DATA============")
    api_manager.generate_data(date)