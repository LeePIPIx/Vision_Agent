from pydantic import BaseModel, Field
from .base_tool import BaseTool
class LesionAnalysisTool(BaseTool):
    name = "lesion_analysis_tool"
    description = "根据空间对齐的 NIfTI 影像和整数标签 mask 计算体积和连通区域；不推断诊断。"
    class Parameters(BaseModel):
        image_path: str = Field(..., description="原始 NIfTI 影像路径")
        mask_path: str = Field(..., description="与影像空间对齐的整数标签 NIfTI mask 路径")
    def run(self, image_path, mask_path):
        from medical_app.imaging import label_statistics
        return label_statistics(image_path, mask_path)
