import json
import os
from datetime import datetime

class JsonResultManager:
    results_path = os.path.join("results")

    def parse_actions(self, action_string):
        if action_string == "No actions" or not action_string:
            return []
        
        actions = []
        action_pairs = action_string.split(", ")
        
        for pair in action_pairs:
            key, value = pair.split(": ")
            actions.append({
                key.strip(): float(value.strip())
            })
        
        return actions

    def dataframe_to_json(self, results_df):
        json_res = {}

        for index, row in results_df.iterrows():
            time_key = row["Current_Hour"]
            json_res[time_key] = {
                "input_data" : {
                    "Solar_Production": row["Solar_Production"],
                    "Wind_Production": row["Wind_Production"],
                    "Consumption": row["Consumption"],
                    "Current_Capacity": row["Current_Capacity"],
                    "Price": row["Price"]
                },
                "Actions": self.parse_actions(row["Actions"]),
                "output_data": {
                    "Balance": row["Balance"],
                    "New_Capacity": row["New_Capacity"]
                }
            }

        return json_res

    def save_to_json_file(self, results_df, agent="smart_agent"):
        filename_json = f"{agent}_{datetime.now():%Y%m%d_%H%M%S}.json"
        filepath_json = os.path.join(self.results_path, agent, filename_json)

        filename_csv = f"{agent}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        filepath_csv = os.path.join(self.results_path, agent, filename_csv)

        results_df.to_csv(filepath_csv)
        json_data = self.dataframe_to_json(results_df)
        
        with open(filepath_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
        
        return True


json_result_manager = JsonResultManager()