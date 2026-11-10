from .base_tool import BaseTool
from pydantic import BaseModel, Field
from tavily import TavilyClient
import os

class WebSearchTool(BaseTool):
    
    name =  "web_search_tool"
    description = "搜索互联网获取信息"
    
    class Parameters(BaseModel):
        query: str = Field(..., description="需要搜索的内容")
        
    def run(self, query: str) -> str:
        """使用Tavily搜索互联网获取信息"""
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = tavily_client.search(query)
        results = response.get("results", [])
        formatted = []
        for r in results:
            formatted.append({"title": r.get("title"), "content": r.get("content"), "url": r.get("url"), "published_date": r.get("published_date")})
        return {"sources": formatted}