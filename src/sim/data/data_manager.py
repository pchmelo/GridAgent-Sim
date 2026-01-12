import os
import pandas as pd
from dotenv import load_dotenv
from log.log_controller import log_controller

try:
    from .api_manager import api_manager
except ImportError:
    from api_manager import api_manager

load_dotenv()

DATE_DEFAULT_DATE = os.getenv("DATE", "2025-01-01")

class DataManager:
    log_type = "simulation"

    def __init__(self, smooth_window=3, date=DATE_DEFAULT_DATE):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.datafiles_dir = os.path.join(self.base_dir, "datafiles")

        self.price_data_filename = "market_prices.csv"
        self.solar_data_filename = "solar_production.csv"
        self.wind_data_filename = "wind_production.csv"
        self.consumption_data_filename = "consumption.csv"
        
        self.smooth_window = smooth_window
        self.start_data_collection(date)

        self.use_api = True

        
    def start_data_collection(self, date: str):
        #log_controller.add_log(f"Starting data collection for date: {date}", self.log_type)
        self.use_api = True

        self.date = date
        self.last_time_stamp = (0, 0)

        date_folder = os.path.join(self.datafiles_dir, date)

        if not os.path.exists(date_folder) or not os.path.isdir(date_folder):
            log_controller.add_log(f"Folder for date {date} does not exist", self.log_type)
            api_manager.generate_data(self.date)
        
        self.get_data_for_date()
        
        '''
        log_controller.add_log(f"Solar production: \n{self.df_solar_production.head()}", self.log_type)
        log_controller.add_log(f"Wind production: \n{self.df_wind_production.head()}", self.log_type)
        log_controller.add_log(f"Market prices: \n{self.df_price_data.head()}", self.log_type)
        log_controller.add_log(f"Consumption: \n{self.df_consumption.head()}", self.log_type)
        '''
        
        return True

    def get_model_data_entry(self, time_stamp: tuple = None, date: str = None):
        if time_stamp is None:
            log_controller.add_log("Time stamp must be provided", self.log_type)
            return None
        
        if (date is not None and (not hasattr(self, 'date') or self.date != date)) and self.use_api:
            log_controller.add_log(f"Date {date} is different from {self.date}", self.log_type)
            self.start_data_collection(date)

        hour, minute = time_stamp

        #log_controller.add_log(f"Time stamp: {time_stamp}", self.log_type)

        price = self.calculate_price_interval(hour, minute)
        solar_production = self.calculate_solar_production_interval(hour, minute)
        wind_production = self.calculate_wind_production_interval(hour, minute)
        consumption = self.calculate_consumption_interval(hour, minute)

        #log_controller.add_log(f"Price: {price}, Solar production: {solar_production}, Wind production: {wind_production}, Consumption: {consumption}", self.log_type)

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
    
    def calculate_market_price_mean(self):
        price_col = self.df_price_data.columns[1]
        mean_price = self.df_price_data[price_col].mean()
    
        return mean_price

    def return_dataframes(self, date: str):
        if date is not None and (not hasattr(self, 'date') or self.date != date):
            self.start_data_collection(date)
        
        return (self.df_price_data, self.df_solar_production,
                self.df_wind_production, self.df_consumption)
    
    def calculate_total_consumption_price(self):
        if hasattr(self, 'df_consumption') and hasattr(self, 'df_price_data'):
            res = 0.0
            
            consumption_col = self.df_consumption.columns[1]
            price_col = self.df_price_data.columns[1]
            
            merged_df = pd.merge(
                self.df_consumption[['total_minutes', consumption_col]], 
                self.df_price_data[['total_minutes', price_col]], 
                on='total_minutes', 
                how='inner'
            )
            
            merged_df['cost'] = merged_df[consumption_col] * merged_df[price_col]
            res = merged_df['cost'].sum()
            
            return True, res
        return False, None
    
    def set_dataframes(self, df_price, df_solar, df_wind, df_consumption):
        self.use_api = False
        self.last_time_stamp = (0, 0)

        self.df_price_data = df_price.copy()
        self.df_solar_production = df_solar.copy()
        self.df_wind_production = df_wind.copy()
        self.df_consumption = df_consumption.copy()
        
        def parse_time_with_24(time_str):
            hour, minute = map(int, time_str.split(':'))
            total_minutes = hour * 60 + minute
            return total_minutes
        
        self.df_solar_production['total_minutes'] = self.df_solar_production.iloc[:, 0].apply(parse_time_with_24)
        self.df_wind_production['total_minutes'] = self.df_wind_production.iloc[:, 0].apply(parse_time_with_24)
        self.df_price_data['total_minutes'] = self.df_price_data.iloc[:, 0].apply(parse_time_with_24)
        self.df_consumption['total_minutes'] = self.df_consumption.iloc[:, 0].apply(parse_time_with_24)


data_manager = DataManager()



if __name__ == "__main__":
    data_manager.start_data_collection("2025-08-25")
    
    price, solar_prod, wind_prod, consumption = data_manager.get_model_data_entry(
        date="2025-08-25", time_stamp="1:30")
    print(f"Price: {price}, Solar: {solar_prod}, Wind: {wind_prod}, Consumption: {consumption}")

    price, solar_prod, wind_prod, consumption = data_manager.get_model_data_entry(
        time_stamp="2:30")  
    print(f"Price: {price}, Solar: {solar_prod}, Wind: {wind_prod}, Consumption: {consumption}")