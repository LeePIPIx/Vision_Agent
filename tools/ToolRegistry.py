from utils.logger import get_logger
from .base_tool import BaseTool

logger = get_logger(__name__)

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, type[BaseTool]] = {}
        
    def register(self, *tool_cls: type[BaseTool]):
        """注册一个工具类，要求继承自 BaseTool"""
        for cls in tool_cls:
            self._tools[cls.name] = cls
            logger.info(f"[ToolRegistry] Registered tool: {cls.name}")
    
    def get(self, name: str) -> type[BaseTool] | None:
        """根据工具名称获取工具类"""
        return self._tools.get(name)
    
    def execute(self, name: str, **kwargs) -> str:
        """执行指定工具，传入参数并返回结果"""
        from medical_app.context import checkpoint, allowed_path
        checkpoint()
        for key in ("file_path", "image_path", "mask_path"):
            if kwargs.get(key):
                kwargs[key] = str(allowed_path(kwargs[key]))
        tool_cls = self._tools.get(name)
        if tool_cls is None:
            logger.error(f"[ToolRegistry] 未知工具: {name}")
            return f"[错误] 未知工具: Tool '{name}' not found."
        try:
            return tool_cls.validate_parameters(**kwargs)
        except Exception as e:
            logger.error("[ToolRegistry] 工具执行异常: %s: %s", name, type(e).__name__)
            return f"[工具执行异常] {name}: {type(e).__name__}: {e}"
        
    @property
    def openai_schema(self) -> list[dict]:
        """返回所有注册工具的 OpenAI schema 列表"""
        return [cls.to_openai_schema() for cls in self._tools.values()]
    
    @property
    def tool_names(self) -> list[str]:
        """返回所有注册工具的名称列表"""
        return list(self._tools.keys())
    