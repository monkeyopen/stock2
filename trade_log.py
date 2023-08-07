
def calculate_profit_loss(initial_investment, current_value):
    profit_loss = current_value - initial_investment
    return profit_loss

def analyze_performance(initial_investment, current_value):
    profit_loss = calculate_profit_loss(initial_investment, current_value)
    percentage_change = (profit_loss / initial_investment) * 100

    if profit_loss > 0:
        performance = "profit"
    elif profit_loss < 0:
        performance = "loss"
    else:
        performance = "break-even"

    return performance, profit_loss, percentage_change

if __name__ == '__main__':
    # 示例
    initial_investment = 1000
    current_value = 1200

    performance, profit_loss, percentage_change = analyze_performance(initial_investment, current_value)
    print(f"Your investment performance is a {performance} of ${profit_loss:.2f}, which is a {percentage_change:.2f}% change.")