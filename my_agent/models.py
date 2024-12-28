from pydantic import BaseModel, Field
from typing import List, Optional


class SubData(BaseModel):
    portfolioId: int
    nseCode: str = Field(..., description="The NSE stock code for each stock.")
    clientName: str
    quantityHeld: str
    holdingPer: str = Field(
        ..., description="The holding percentage for the stock in the portfolio."
    )
    changePrev: str = Field(
        ...,
        description=(
            "The change in holding percentage compared to the previous quarter. "
            "If `changePrev` is 0, mention it as 'No change'. "
            "If `changePrev` is 'New', highlight it in the output as 'New'."
        ),
    )
    holdingVal: str
    displayLock: str


class DataListItem(BaseModel):
    portfolioId: int
    nseCode: str = Field(..., description="The NSE stock code for each stock.")
    scDid: str
    scId: str
    stockName: str
    holderName: str
    quantityHeld: str
    holdingPer: str = Field(
        ..., description="The holding percentage for the stock in the portfolio."
    )
    changePrev: str = Field(
        ...,
        description=(
            "The change in holding percentage compared to the previous quarter. "
            "If `changePrev` is 0, mention it as 'No change'. "
            "If `changePrev` is 'New', highlight it in the output as 'New'."
        ),
    )
    holdingVal: str
    url: str
    quarter: str
    changePrevClass: str
    subCount: int
    subData: Optional[List[SubData]]
    displayLock: str
    exchange: str


class PortfolioHolding(BaseModel):
    success: int
    data: dict
