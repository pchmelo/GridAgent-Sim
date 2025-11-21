import json
import os
from datetime import datetime
from sim.data.data_manager import data_manager

class JsonResultManager:
    results_path = os.path.join(os.path.dirname(__file__), "results")
    final_json_data = {}
    final_json_filename = f"final_results_{datetime.now():%Y%m%d_%H%M%S}.json"
    final_json_filepath = os.path.join(results_path, "final_results", final_json_filename)

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

        self.final_json_data[self.agent_type] = json_res

        return json_res

    def save_to_json_file(self, results_df, agent_type="smart"):
        self.agent_type = agent_type

        filename_json = f"{agent_type}_{datetime.now():%Y%m%d_%H%M%S}.json"
        filepath_json = os.path.join(self.results_path, "json", agent_type, filename_json)

        filename_csv = f"{agent_type}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        filepath_csv = os.path.join(self.results_path, "csv", agent_type, filename_csv)
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(filepath_csv), exist_ok=True)
        os.makedirs(os.path.dirname(filepath_json), exist_ok=True)
        
        results_df.to_csv(filepath_csv)
        json_data = self.dataframe_to_json(results_df)
        
        with open(filepath_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
        
        return True
    
    def calculate_final_results(self):
        if self.final_json_data["smart"] and self.final_json_data["basic"]:
            res = {}
            smart_last_key = list(self.final_json_data["smart"].keys())[-1]
            basic_last_key = list(self.final_json_data["basic"].keys())[-1]
            
            smart_balance = self.final_json_data["smart"][smart_last_key]["output_data"]["Balance"]
            basic_balance = self.final_json_data["basic"][basic_last_key]["output_data"]["Balance"]

            res["smart_agent_balance"] = smart_balance
            res["basic_agent_balance"] = basic_balance
            res["agent_balance_difference"] = smart_balance - basic_balance

            flag, total_consumption_cust = data_manager.calculate_total_consumption_price()
            if flag:
                res["total_consumption_cust"] = 0 - total_consumption_cust
                res["basic_agent_consumption_saving"] = abs(basic_balance - (0 -total_consumption_cust))
                res["smart_agent_consumption_saving"] = abs(smart_balance - (0 -total_consumption_cust))

            self.final_json_data["final_results"] = res

            os.makedirs(os.path.dirname(self.final_json_filepath), exist_ok=True)

            with open(self.final_json_filepath, 'w', encoding='utf-8') as f:
                json.dump(self.final_json_data, f, indent=4, ensure_ascii=False)

            return True
        return False


json_result_manager = JsonResultManager()
