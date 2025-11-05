import os
import pandas as pd
from dotenv import load_dotenv

try:
    from .api_manager import api_manager
except ImportError:
    from api_manager import api_manager

load_dotenv()

DATE_DEFAULT_DATE = os.getenv("DATE", "2023-01-01")

class DataManager:
    def __init__(self, smooth_window=3, date=DATE_DEFAULT_DATE):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.datafiles_dir = os.path.join(self.base_dir, "datafiles")

        self.price_data_filename = "market_prices.csv"
        self.solar_data_filename = "solar_production.csv"
        self.wind_data_filename = "wind_production.csv"
        self.consumption_data_filename = "consumption.csv"
        
        self.smooth_window = smooth_window
        self.start_data_collection(date)

        
    def start_data_collection(self, date: str):
        self.date = date
        self.last_time_stamp = (0, 0)

        date_folder = os.path.join(self.datafiles_dir, date)

        if not os.path.exists(date_folder) or not os.path.isdir(date_folder):
            print(f"Folder for date {date} does not exist")
            api_manager.generate_data(self.date)
        
        self.get_data_for_date()
        """
        print(f"Solar production: \n{self.df_solar_production.head()}")
        print(f"Wind production: \n{self.df_wind_production.head()}")
        print(f"Market prices: \n{self.df_price_data.head()}")
        print(f"Consumption: \n{self.df_consumption.head()}")
        """
        return True

    def get_model_data_entry(self, time_stamp: tuple = None, date: str = None):
        if time_stamp is None:
            raise ValueError("Time stamp must be provided")
        
        if date is not None and (not hasattr(self, 'date') or self.date != date):
            self.start_data_collection(date)

        hour, minute = time_stamp

        price = self.calculate_price_interval(hour, minute)
        solar_production = self.calculate_solar_production_interval(hour, minute)
        wind_production = self.calculate_wind_production_interval(hour, minute)
        consumption = self.calculate_consumption_interval(hour, minute)

        return price, solar_production, wind_production, consumption

    def update_time_stamp(self, new_time_stamp: tuple):
        self.last_time_stamp = new_time_stamp

    def get_data_for_date(self):
        self.df_solar_production = pd.read_csv(os.path.join(
            self.datafiles_dir, self.date, self.solar_data_filename))
        self.df_wind_production = pd.read_csv(os.path.join(
            self.datafiles_dir, self.date, self.wind_data_filename))    
        self.df_price_data = pd.read_csv(os.path.join(
            self.datafiles_dir, self.date, self.price_data_filename))
        self.df_consumption = pd.read_csv(os.path.join(
            self.datafiles_dir, self.date, self.consumption_data_filename))
        
        def parse_time_with_24(time_str):
            hour, minute = map(int, time_str.split(':'))
            total_minutes = hour * 60 + minute
            return total_minutes
        
        self.df_solar_production['total_minutes'] = self.df_solar_production.iloc[:, 0].apply(parse_time_with_24)
        self.df_wind_production['total_minutes'] = self.df_wind_production.iloc[:, 0].apply(parse_time_with_24)
        self.df_price_data['total_minutes'] = self.df_price_data.iloc[:, 0].apply(parse_time_with_24)
        self.df_consumption['total_minutes'] = self.df_consumption.iloc[:, 0].apply(parse_time_with_24)
    
    def calculate_solar_production_interval(self, new_hour: int, new_minute: int):
        last_hour, last_minute = self.last_time_stamp
        
        last_total_minutes = last_hour * 60 + last_minute
        new_total_minutes = new_hour * 60 + new_minute
        
        if new_total_minutes < last_total_minutes:
            new_total_minutes += 1440
        
        interval_hours = (new_total_minutes - last_total_minutes) / 60
        
        if new_total_minutes > 1440:
            mask1 = (self.df_solar_production['total_minutes'] > last_total_minutes) & \
                   (self.df_solar_production['total_minutes'] <= 1440)
            interval_data1 = self.df_solar_production[mask1]
            
            remainder_minutes = new_total_minutes - 1440
            mask2 = (self.df_solar_production['total_minutes'] > 0) & \
                   (self.df_solar_production['total_minutes'] <= remainder_minutes)
            interval_data2 = self.df_solar_production[mask2]
            
            interval_data = pd.concat([interval_data1, interval_data2])
        else:
            mask = (self.df_solar_production['total_minutes'] > last_total_minutes) & \
                  (self.df_solar_production['total_minutes'] <= new_total_minutes)
            interval_data = self.df_solar_production[mask]
        
        production_col = self.df_solar_production.columns[1]
        
        if len(interval_data) > 0:
            avg_power = interval_data[production_col].mean()
        else:
            all_minutes = sorted(self.df_solar_production['total_minutes'].unique())
            max_minute = max(all_minutes)
            min_minute = min(all_minutes)
            
            clamped_last = max(min_minute, min(last_total_minutes, max_minute))
            clamped_new = max(min_minute, min(new_total_minutes % 1440, max_minute))
            
            indexed_production = self.df_solar_production.set_index('total_minutes')[production_col]
            reindexed = indexed_production.reindex(all_minutes).interpolate(method='linear')
            
            start_value = reindexed.reindex([clamped_last], method='nearest')[clamped_last]
            end_value = reindexed.reindex([clamped_new], method='nearest')[clamped_new]
            
            avg_power = (start_value + end_value) / 2
        
        energy_production = avg_power * interval_hours
        
        return energy_production

    def calculate_wind_production_interval(self, new_hour: int, new_minute: int):
        last_hour, last_minute = self.last_time_stamp
        
        last_total_minutes = last_hour * 60 + last_minute
        new_total_minutes = new_hour * 60 + new_minute
        
        if new_total_minutes < last_total_minutes:
            new_total_minutes += 1440
        
        interval_hours = (new_total_minutes - last_total_minutes) / 60
        
        if new_total_minutes > 1440:
            mask1 = (self.df_wind_production['total_minutes'] > last_total_minutes) & \
                   (self.df_wind_production['total_minutes'] <= 1440)
            interval_data1 = self.df_wind_production[mask1]
            
            remainder_minutes = new_total_minutes - 1440
            mask2 = (self.df_wind_production['total_minutes'] > 0) & \
                   (self.df_wind_production['total_minutes'] <= remainder_minutes)
            interval_data2 = self.df_wind_production[mask2]
            
            interval_data = pd.concat([interval_data1, interval_data2])
        else:
            mask = (self.df_wind_production['total_minutes'] > last_total_minutes) & \
                  (self.df_wind_production['total_minutes'] <= new_total_minutes)
            interval_data = self.df_wind_production[mask]
        
        production_col = self.df_wind_production.columns[1]
        
        if len(interval_data) > 0:
            avg_power = interval_data[production_col].mean()
        else:
            all_minutes = sorted(self.df_wind_production['total_minutes'].unique())
            max_minute = max(all_minutes)
            min_minute = min(all_minutes)
            
            clamped_last = max(min_minute, min(last_total_minutes, max_minute))
            clamped_new = max(min_minute, min(new_total_minutes % 1440, max_minute))
            
            indexed_production = self.df_wind_production.set_index('total_minutes')[production_col]
            reindexed = indexed_production.reindex(all_minutes).interpolate(method='linear')
            
            start_value = reindexed.reindex([clamped_last], method='nearest')[clamped_last]
            end_value = reindexed.reindex([clamped_new], method='nearest')[clamped_new]
            
            avg_power = (start_value + end_value) / 2
        
        energy_production = avg_power * interval_hours
        
        return energy_production

    def calculate_consumption_interval(self, new_hour: int, new_minute: int):
        last_hour, last_minute = self.last_time_stamp
        
        last_total_minutes = last_hour * 60 + last_minute
        new_total_minutes = new_hour * 60 + new_minute
        
        if new_total_minutes < last_total_minutes:
            new_total_minutes += 1440
        
        interval_hours = (new_total_minutes - last_total_minutes) / 60
        
        if new_total_minutes > 1440:
            mask1 = (self.df_consumption['total_minutes'] > last_total_minutes) & \
                   (self.df_consumption['total_minutes'] <= 1440)
            interval_data1 = self.df_consumption[mask1]
            
            remainder_minutes = new_total_minutes - 1440
            mask2 = (self.df_consumption['total_minutes'] > 0) & \
                   (self.df_consumption['total_minutes'] <= remainder_minutes)
            interval_data2 = self.df_consumption[mask2]
            
            interval_data = pd.concat([interval_data1, interval_data2])
        else:
            mask = (self.df_consumption['total_minutes'] > last_total_minutes) & \
                  (self.df_consumption['total_minutes'] <= new_total_minutes)
            interval_data = self.df_consumption[mask]
        
        consumption_col = self.df_consumption.columns[1]
        
        if len(interval_data) > 0:
            avg_power = interval_data[consumption_col].mean()
        else:
            all_minutes = sorted(self.df_consumption['total_minutes'].unique())
            max_minute = max(all_minutes)
            min_minute = min(all_minutes)
            
            clamped_last = max(min_minute, min(last_total_minutes, max_minute))
            clamped_new = max(min_minute, min(new_total_minutes % 1440, max_minute))
            
            indexed_consumption = self.df_consumption.set_index('total_minutes')[consumption_col]
            reindexed = indexed_consumption.reindex(all_minutes).interpolate(method='linear')
            
            start_value = reindexed.reindex([clamped_last], method='nearest')[clamped_last]
            end_value = reindexed.reindex([clamped_new], method='nearest')[clamped_new]
            
            avg_power = (start_value + end_value) / 2
        
        energy_consumption = avg_power * interval_hours
        
        return energy_consumption

    def calculate_price_interval(self, new_hour: int, new_minute: int):
        last_hour, last_minute = self.last_time_stamp
        
        last_total_minutes = last_hour * 60 + last_minute
        new_total_minutes = new_hour * 60 + new_minute
        
        if new_total_minutes < last_total_minutes:
            new_total_minutes += 1440
        
        if new_total_minutes > 1440:
            mask1 = (self.df_price_data['total_minutes'] > last_total_minutes) & \
                   (self.df_price_data['total_minutes'] <= 1440)
            interval_data1 = self.df_price_data[mask1]
            
            remainder_minutes = new_total_minutes - 1440
            mask2 = (self.df_price_data['total_minutes'] > 0) & \
                   (self.df_price_data['total_minutes'] <= remainder_minutes)
            interval_data2 = self.df_price_data[mask2]
            
            interval_data = pd.concat([interval_data1, interval_data2])
        else:
            mask = (self.df_price_data['total_minutes'] > last_total_minutes) & \
                  (self.df_price_data['total_minutes'] <= new_total_minutes)
            interval_data = self.df_price_data[mask]
        
        price_col = self.df_price_data.columns[1]
        
        if len(interval_data) > 0:
            mean_price = interval_data[price_col].mean()
        else:
            all_minutes = sorted(self.df_price_data['total_minutes'].unique())
            max_minute = max(all_minutes)
            min_minute = min(all_minutes)
            
            clamped_last = max(min_minute, min(last_total_minutes, max_minute))
            clamped_new = max(min_minute, min(new_total_minutes % 1440, max_minute))
            
            indexed_prices = self.df_price_data.set_index('total_minutes')[price_col]
            
            reindexed = indexed_prices.reindex(all_minutes).interpolate(method='linear')
            
            start_value = reindexed.reindex([clamped_last], method='nearest')[clamped_last]
            end_value = reindexed.reindex([clamped_new], method='nearest')[clamped_new]
            
            mean_price = (start_value + end_value) / 2

        return mean_price

data_manager = DataManager()


if __name__ == "__main__":
    data_manager.start_data_collection("2025-08-25")
    
    price, solar_prod, wind_prod, consumption = data_manager.get_model_data_entry(
        date="2025-08-25", time_stamp="1:30")
    print(f"Price: {price}, Solar: {solar_prod}, Wind: {wind_prod}, Consumption: {consumption}")

    price, solar_prod, wind_prod, consumption = data_manager.get_model_data_entry(
        time_stamp="2:30")  
    print(f"Price: {price}, Solar: {solar_prod}, Wind: {wind_prod}, Consumption: {consumption}")