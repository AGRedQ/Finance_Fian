import requests

def get_yahoo_finance_news(query="Apple", count=10):
    url = f"https://query2.finance.yahoo.com/v1/finance/search"
    params = {
        "q": query,
        "newsCount": count,
        "quotesCount": 0,
        "listsCount": 0,
        "enableFuzzyQuery": False,
        "enableCb": False,
        "enableNavLinks": False,
        "enableEnhancedTrivialQuery": False
    }
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    news_items = data.get("news", [])
    news_list = []
    for item in news_items:
        news_list.append({
            "title": item.get("title"),
            "publisher": item.get("publisher"),
            "link": item.get("link"),
            "providerPublishTime": item.get("providerPublishTime")
        })
    return news_list

if __name__ == "__main__":
    news = get_yahoo_finance_news("Apple", 10)
    for n in news:
        print(f"{n['title']} ({n['publisher']})")
        print(f"Link: {n['link']}\n")