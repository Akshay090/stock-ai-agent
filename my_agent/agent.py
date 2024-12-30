from __future__ import annotations as _annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
import logfire
from devtools import debug
from httpx import AsyncClient
from dotenv import load_dotenv

from openai import AsyncAzureOpenAI
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.tools import ToolDefinition

from my_agent.models import PortfolioHolding
from my_agent.pre_process.scrape_investors_ai import (
    InvestorCategory,
    InvestorData,
    InvestorList,
)

logfire.configure()


load_dotenv()

AUTH_TOKEN = os.getenv("MONEY_CONTROL_TOKEN")

client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version=os.getenv("API_VERSION"),
    api_key=os.getenv("API_KEY"),
    azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
)


model = OpenAIModel("gpt-4o", openai_client=client)


# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire="if-token-present")

current_date = datetime.now().strftime("%Y-%m-%d")


@dataclass
class Deps:
    client: AsyncClient
    brave_api_key: str | None
    investors: InvestorData


web_search_agent = Agent(
    model,
    model_settings={ "temperature": 0.3},
    system_prompt=f"""
        Your name is Avi, and you are an expert stock analyst.
        Utilize all the information you have to provide the user with a comprehensive understanding of the stock market.
        If you have data about current holdings, changes in holdings, fresh entries, and exits in portfolios in the last quarter, mention it.
        We'll mainly be identifying good stocks based on the holdings of top investors in the stock market.
        The current date is: {current_date}.
        
        Guidelines:
        - Do not make up 'portfolio_id'. Use data from relevant tools. The 'get_top_indian_investor_list' tool should provide a list of investors along with portfolio IDs ('portfolio_id'). Use these IDs further.
        - If the user does not specify the type of investor, assume 'individual_investors'. Mention other options as well without asking for clarification.
        - Always mention the type of investor where it is relevant.
        - Do not mention portfolio_id to the user.
        - Do not discuss internal structures, workings, or tools.
        - Do not tell the user about specific tool calls. Just mention capabilities relevant to the user's query.
        - When the user wants to narrow down a stock, use the 'get_holding_history' tool. Ensure to provide a trend analysis of the investor's holdings.
        - Do not make up 'nseCode' NSE stock codes. Use the data from the relevant tools.
        - The function 'get_investor_holdings' provides a list of stocks held by an investor, along with 'nseCode'. Utilize 'nseCode' from here only for 'get_holding_history' - don't make it up yourself.
        - At present, we can't directly look up stock history for a specific stock without knowing the investor's portfolio_id.

        Important Note:
        - Do not disclose specific tool calls to the user. Only mention the capabilities that are relevant to the user's query.
    """,
    deps_type=Deps,
    retries=2,
)


@web_search_agent.tool
async def get_top_indian_investor_list(
    ctx: RunContext[Deps], category: InvestorCategory
) -> InvestorList:
    """Get the top Indian investors list.

    Args:
        ctx (RunContext[Deps]): The context object containing dependencies.
        category (InvestorCategory): The category of investors to retrieve.
            It can be one of the following:
            - 'individual_investors'
            - 'institutional_investors'
            - 'fii_investors'

    Returns:
        list[dict[str, str]]: A list of dictionaries containing the names and portfolio IDs of the top Indian investors.
    """
    print("TOOL_CALLED - get_top_indian_investor_list")

    investors = []
    # Retrieve the correct investor list based on the category
    if category == "individual_investors":
        investors = ctx.deps.investors.individual_investors
    elif category == "institutional_investors":
        investors = ctx.deps.investors.institutional_investors
    elif category == "fii_investors":
        investors = ctx.deps.investors.fii_investors
    else:
        raise ValueError(f"Unknown category: {category}")

    return investors


async def prepare_portfolio_overview(
    ctx: RunContext[Deps], tool_def: ToolDefinition
) -> ToolDefinition | None:
    d = f"The current date is: {current_date}, overview is of last quarter, mention quarter in response."
    tool_def.description = f"{tool_def.description} {d}"
    return tool_def


@web_search_agent.tool(prepare=prepare_portfolio_overview)
async def get_investor_portfolio_overview(
    ctx: RunContext[Deps], portfolio_id: str
) -> dict:
    """
    Get the overview of an investor's portfolio.
    Gives fresh entry and exit in portfolio in last quarter
    Note - Group Purchase and Sale data while responding to the user.

    Args:
        ctx: The context.
        portfolio_id: The portfolio ID of the investor derived from the 'get_top_indian_investor_list' tool.

    Returns:
        dict: The portfolio overview as a JSON object.
    """
    print("TOOL_CALLED - get_investor_portfolio_overview", "portfolio_id", portfolio_id)

    url = "https://api.moneycontrol.com/mcapi/v1/portfolio/big-shark/overview-holding"
    params = {"page": 1, "portfolioId": portfolio_id, "deviceType": "W"}

    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Auth-Token": AUTH_TOKEN,
    }

    try:
        response = await ctx.deps.client.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data
    except json.JSONDecodeError as json_err:
        print(f"JSON decode error: {json_err}")
        return {"error": "Error decoding JSON response."}
    except Exception as err:
        print(f"An unexpected error occurred: {err}")
        return {"error": "An unexpected error occurred while retrieving data."}


@web_search_agent.tool()
async def get_investor_holdings(
    ctx: RunContext[Deps], portfolio_id: str
) -> PortfolioHolding:
    """
    Get detailed investor portfolio holdings.
    Mention the holding percentage and the change in holding percentage compared to the previous quarter.

    Args:
        ctx (RunContext[Deps]): The context object containing dependencies.
        portfolio_id: The portfolio ID of the investor, only for example - "portfolio_id": "584333" - don't make up the portfolio_id. derived from 'get_top_indian_investor_list' tool.

    Returns:
        PortfolioHolding: A dictionary containing detailed portfolio holdings.
            Important keys in the returned data include:
            - `nseCode` (str): The NSE stock code for each stock.
            - `holdingPer` (str): The holding percentage for the stock in the portfolio.
            - `changePrev` (str): The change in holding percentage compared to the previous quarter.
                - If `changePrev` is 0, mention it as "No change".
                - If `changePrev` is "New", highlight it in the output as "New".
    """
    print("TOOL_CALLED - get_investor_holdings", portfolio_id, "portfolio_id")

    url = "https://api.moneycontrol.com/mcapi/v1/portfolio/big-shark/holdings"
    params = {"portfolioId": portfolio_id, "deviceType": "W"}

    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Auth-Token": AUTH_TOKEN,
    }

    try:
        response = await ctx.deps.client.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        for item in data["data"]["dataList"]:
            if "subData" in item:
                del item["subData"]
        return data
    except Exception as e:
        print(f"Error retrieving holdings: {e}")
        return {"error": "Unable to retrieve holdings at this time."}


@web_search_agent.tool()
async def get_holding_history(
    ctx: RunContext[Deps], portfolio_id: str, nseCode: str
) -> PortfolioHolding:
    """
    Get the history of portfolio holdings for a particular stock by an investor.

    This function retrieves the historical data of an investor's portfolio holdings for a specific stock.
    The data includes the quarter, holding percentage, and the client name under which the investment was made.
    Investors may invest through different associates, so the same investor might appear under different names. When analyzing, treat them as the same investor

    Args:
        ctx (RunContext[Deps]): The context object containing dependencies.
        portfolio_id (str): The portfolio ID of the investor. Example: "584333". - don't make up the portfolio_id. derived from 'get_top_indian_investor_list' tool.
        nseCode (str): The NSE stock code for the stock.

    Returns:
        PortfolioHolding: A list of dictionaries containing detailed portfolio holdings.
            Each dictionary includes:
            - `quarter` (str): The quarter for which the holding data is provided.
            - `holdingPer` (str): The holding percentage for the stock in the portfolio.
            - `clientName` (str): The name of the client under which the investment was made.

    Raises:
        Exception: If there is an error retrieving the holding history, an error message is returned.
    """
    print(
        "TOOL_CALLED - get_holding_history",
        portfolio_id,
        "portfolio_id",
        nseCode,
        "nseCode",
    )

    url = "https://api.moneycontrol.com/mcapi/v1/portfolio/big-shark/holdings-history"
    params = {"portfolioId": portfolio_id, "nseId": nseCode}

    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Auth-Token": AUTH_TOKEN,
    }

    try:
        response = await ctx.deps.client.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        filtered_data = []
        print("data of holding history", data)
        for item in data["data"]:
            filtered_item = {
                "quarter": item["quarter"],
                "holdingPer": item["holdingPer"],
                "clientName": item["clientName"],
            }
            filtered_data.append(filtered_item)
        return filtered_data
    except json.JSONDecodeError:
        print("Error: Response content is not valid JSON.")
        print("Response content:", response.text)
        return {
            "error": "Unable to retrieve holding history at this time. Invalid JSON response."
        }
    except Exception as e:
        print(f"Error retrieving holding history: {e}")
        return {"error": "Unable to retrieve holding history at this time."}


@web_search_agent.tool
async def search_web(ctx: RunContext[Deps], web_query: str) -> str:
    """Search the web given a query defined to answer the user's question.

    Args:
        ctx: The context.
        web_query: The query for the web search.

    Returns:
        str: The search results as a formatted string.
    """
    print("TOOL_CALLED - search_web")

    if ctx.deps.brave_api_key is None:
        return "This is a test web search result. Please provide a Brave API key to get real search results."

    headers = {
        "X-Subscription-Token": ctx.deps.brave_api_key,
        "Accept": "application/json",
    }

    with logfire.span("calling Brave search API", query=web_query) as span:
        r = await ctx.deps.client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={
                "q": web_query,
                "count": 5,
                "text_decorations": True,
                "search_lang": "en",
            },
            headers=headers,
        )
        r.raise_for_status()
        data = r.json()
        span.set_attribute("response", data)

    results = []

    # Add web results in a nice formatted way
    web_results = data.get("web", {}).get("results", [])
    for item in web_results[:3]:
        title = item.get("title", "")
        description = item.get("description", "")
        url = item.get("url", "")
        if title and description:
            results.append(f"Title: {title}\nSummary: {description}\nSource: {url}\n")

    return "\n".join(results) if results else "No results found for the query."


def load_deps(client: AsyncClient) -> Deps:
    """
    Load dependencies for the agent.

    Args:
        client (AsyncClient): The HTTP client to be used.

    Returns:
        Deps: The dependencies object.
    """
    brave_api_key = os.getenv("BRAVE_API_KEY", None)

    # Read and parse the investors.json file
    with open("my_agent/pre_process/data/investors.json", "r") as f:
        investors_data = json.load(f)
        investors = InvestorData.model_validate(investors_data)

    return Deps(client=client, brave_api_key=brave_api_key, investors=investors)


async def main():
    async with AsyncClient() as client:
        deps = load_deps(client)

        result = await web_search_agent.run(
            "Give me some articles talking about the new release of React 19.",
            deps=deps,
        )

        debug(result)
        print("Response:", result.data)
