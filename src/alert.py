def generate_alert(probability):
    """
    Generate alert message
    """

    prob = probability[0][1]

    if prob > 0.7:
        return "🔴 HIGH RISK"
    elif prob > 0.4:
        return "🟡 WARNING"
    else:
        return "🟢 NORMAL"