from math import e
import os
from dotenv import load_dotenv
from datetime import datetime

class LogController:
    colors = {
        "baseline_input": "\033[91m",
        "action_validation": "\033[94m",               
    }

    type_mapping = {
        "baseline_input": "BASELINE INPUT",
        "action_validation": "ACTION VALIDATION",
    }

    reset = "\033[0m"  

    def __init__(self):
        load_dotenv()
        self.log_active = os.getenv("LOG_ACTIVE", "TRUE") == "TRUE"
        
        self.baseline_input = os.getenv("BASELINE_INPUT", "FALSE") == "TRUE"
        self.action_validation = os.getenv("ACTION_VALIDATION", "FALSE") == "TRUE"

        current_date = datetime.now().strftime("%Y-%m-%d")
        log_folder = os.path.join(os.path.dirname(__file__), "files")
        self.log_file = os.path.join(log_folder, f"log_{current_date}.txt")

        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as file:
                file.write("# Log file created on " + current_date + "\n")

    
    def log_message(self, message, type):
        if self.log_active:
            if self.baseline_input:
                pass
            elif self.action_validation:
                pass
            else:
                return
            self.add_log(message, type)

    def add_log(self, message, type):
        timestamp = datetime.now().strftime("%H:%M:%S")  

        type_str = self.type_mapping.get(type, "UNKNOWN")
        yellow_timestamp = f"\033[93m{timestamp}\033[0m"
        
        # Get color for type, default to white if not found
        color_type = self.colors.get(type, "\033[0m")

        with open(self.log_file, "a") as file:
            file.write(f"[{timestamp}] [{type_str}]: {message}\n")
            print(f"[{yellow_timestamp}] [{color_type}{type_str}{self.reset}]: {message}")

    
    def get_last_log_file_path(self):
        try:
            log_folder = os.path.join(os.path.dirname(__file__), "files")
            
            if not os.path.exists(log_folder):
                return None
            
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_log_file = os.path.join(log_folder, f"log_{current_date}.txt")
            
            if os.path.exists(current_log_file):
                return current_log_file
            
            log_files = []
            for filename in os.listdir(log_folder):
                if filename.startswith("log_") and filename.endswith(".txt"):
                    file_path = os.path.join(log_folder, filename)
                    if os.path.isfile(file_path):
                        try:
                            date_part = filename[4:-4]  
                            file_date = datetime.strptime(date_part, "%Y-%m-%d")
                            log_files.append((file_date, file_path))
                        except ValueError:
                            continue
            
            if log_files:
                log_files.sort(key=lambda x: x[0], reverse=True)
                return log_files[0][1]  
            
            return None
            
        except Exception as e:
            print(f"Error getting last log file path: {e}")
            return None
        

log_controller = LogController()

if __name__ == "__main__":
    log_controller.log_message("This is a test log message for baseline input.", "baseline_input")