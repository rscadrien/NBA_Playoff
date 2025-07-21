#Function for getting valid input
def get_valid_input_str(prompt: str, valid_options: list[str]) -> str:
    while True:
        value = input(prompt)
        if value in valid_options:
            return value
        else:
            print(f"❌ Invalid input. Please enter one of: {', '.join(valid_options)}")

def get_valid_input_seed(prompt: str,tot):
    while True:
        value = input(prompt)
        value_int=int(value)
        if 1 <= value_int <= tot:
            return value_int
        else:
            print(f"Invalid input. Please enter a value between 1 and 30")

def get_valid_input_record(prompt: str):
    while True:
        value = input(prompt)
        value_fl=float(value)
        if 0 <= value_fl <= 1:
            return value_fl
        else:
            print(f"Invalid input. Please enter a value between 0 and 1")