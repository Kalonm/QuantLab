from datetime import timedelta

def tf_to_timedelta(tf_name: str) -> timedelta:
    # minutes
    if tf_name == "M5":
        return timedelta(minutes=5)
    if tf_name == "M10":
        return timedelta(minutes=10)
    if tf_name == "M15":
        return timedelta(minutes=15)
    if tf_name == "M20":
        return timedelta(minutes=20)
    if tf_name == "M30":
        return timedelta(minutes=30)

    # hours
    if tf_name == "H1":
        return timedelta(hours=1)
    if tf_name == "H2":
        return timedelta(hours=2)
    if tf_name == "H3":
        return timedelta(hours=3)
    if tf_name == "H4":
        return timedelta(hours=4)
    if tf_name == "H6":
        return timedelta(hours=6)
    if tf_name == "H8":
        return timedelta(hours=8)
    if tf_name == "H12":
        return timedelta(hours=12)

    # days / weeks / months
    if tf_name == "D1":
        return timedelta(days=1)
    if tf_name == "W1":
        return timedelta(weeks=1)
    if tf_name == "MN1":
        # crude month approximation; MT5 bar boundaries handle the rest
        return timedelta(days=30)

    raise ValueError(f"Unknown timeframe name: {tf_name}")
