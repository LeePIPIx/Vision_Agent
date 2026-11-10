from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, ValidationError

class BaseTool(ABC):

    name: str
    description: str
    
    class Parameters(BaseModel):
        # 工具输入参数
        pass
    
    @abstractmethod
    def run(self, **kwargs) -> str:
        # 执行工具的逻辑
        pass
    
    # `@classmethod`装饰器用于定义类方法，类方法的第一个参数是cls，表示类本身，而不是实例。
    @classmethod   
    def to_openai_schema(cls) -> dict:
        return {
            "type":"function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": cls.Parameters.model_json_schema()
            }
        }
    
    @classmethod
    def validate_parameters(cls, **kwargs) -> str:
        instance = cls()
        try:
            validated = cls.Parameters(**kwargs)
        except ValidationError as e:
            return f"[参数验证失败] {cls.name} :{e}"
        return instance.run(**validated.model_dump())
            
    