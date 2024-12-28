from __future__ import annotations as _annotations

from dataclasses import dataclass
from datetime import datetime
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


web_search_agent = Agent(
    model,
    system_prompt=f"""
        Your name is Avi, you're a expert stock analyst.
        Utilize all the info you have to give user full idea about the stock market.
        If you have data about current holding, change in holding, fresh entry and exit in portfolio in last quarter, mention it.
        The current date is: {current_date}
    """,
    deps_type=Deps,
    retries=2,
)


@web_search_agent.tool_plain
async def get_top_indian_investor_list() -> list[dict[str, str]]:
    """Get the top Indian investors list."""
    investors = [
        {"name": "Vijay Kishanlal Kedia", "portfolioId": 53805},
        {"name": "Rakesh Jhunjhunwala", "portfolioId": 53781},
    ]
    return investors


async def prepare_portfolio_overview(
    ctx: RunContext[Deps], tool_def: ToolDefinition
) -> ToolDefinition | None:
    d = f"The current date is: {current_date}, overview is of last quarter, mention quarter in response."
    tool_def.description = f"{tool_def.description} {d}"
    return tool_def


@web_search_agent.tool(prepare=prepare_portfolio_overview)
async def get_investor_portfolio_overview(
    ctx: RunContext[Deps], portfolio_id: int
) -> dict:
    """
    Get the overview of an investor's portfolio.
    Gives fresh entry and exit in portfolio in last quarter
    Note - Group Purchase and Sale data while responding to the user.

    Args:
        ctx: The context.
        portfolio_id: The portfolio ID of the investor.

    Returns:
        dict: The portfolio overview as a JSON object.
    """
    url = "https://api.moneycontrol.com/mcapi/v1/portfolio/big-shark/overview-holding"
    params = {"page": 1, "portfolioId": portfolio_id, "deviceType": "W"}

    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Auth-Token": AUTH_TOKEN,
    }

    response = await ctx.deps.client.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()
    return data


@web_search_agent.tool()
async def get_investor_holdings(
    ctx: RunContext[Deps], portfolio_id: int
) -> PortfolioHolding:
    """
    Get detailed investor portfolio holdings.
    Mention the holding percentage and the change in holding percentage compared to the previous quarter.

    Args:
        ctx (RunContext[Deps]): The context object containing dependencies.
        portfolio_id (int): The unique identifier for the investor's portfolio.

    Returns:
        PortfolioHolding: A dictionary containing detailed portfolio holdings.
            Important keys in the returned data include:
            - `nseCode` (str): The NSE stock code for each stock.
            - `holdingPer` (str): The holding percentage for the stock in the portfolio.
            - `changePrev` (str): The change in holding percentage compared to the previous quarter.
                - If `changePrev` is 0, mention it as "No change".
                - If `changePrev` is "New", highlight it in the output as "New".
    """

    url = "https://api.moneycontrol.com/mcapi/v1/portfolio/big-shark/holdings"
    params = {"portfolioId": portfolio_id, "deviceType": "W"}

    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Auth-Token": AUTH_TOKEN,
    }

    response = await ctx.deps.client.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()
    for item in data["data"]["dataList"]:
        if "subData" in item:
            del item["subData"]
    return data


@web_search_agent.tool
async def search_web(ctx: RunContext[Deps], web_query: str) -> str:
    """Search the web given a query defined to answer the user's question.

    Args:
        ctx: The context.
        web_query: The query for the web search.

    Returns:
        str: The search results as a formatted string.
    """
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


async def main():
    async with AsyncClient() as client:
        brave_api_key = os.getenv("BRAVE_API_KEY", None)
        deps = Deps(client=client, brave_api_key=brave_api_key)

        result = await web_search_agent.run(
            "Give me some articles talking about the new release of React 19.",
            deps=deps,
        )

        debug(result)
        print("Response:", result.data)
