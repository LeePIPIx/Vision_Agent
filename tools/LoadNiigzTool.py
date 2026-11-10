from pydantic import BaseModel, Field
from .base_tool import BaseTool
class LoadNiigzTool(BaseTool):
    name = "load_niigz"
    description = "读取 NIfTI 影像尺寸、空间方向、体素间距、单位及校验摘要。"
    class Parameters(BaseModel):
        file_path: str = Field(..., description="NIfTI 文件路径")
    def run(self, file_path):
        from medical_app.imaging import metadata
        return metadata(file_path)
