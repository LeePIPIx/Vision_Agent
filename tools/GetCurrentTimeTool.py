from .base_tool import BaseTool
from pydantic import BaseModel, Field
from datetime import datetime

class GetCurrentTimeTool(BaseTool):
    name = "get_current_time"
    description = '获取当前时间'
    
    class Parameters(BaseModel):
        pass 
    
    def run(self) -> str:
        """获取当前时间"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        