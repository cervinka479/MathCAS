from datetime import timedelta

def format_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))