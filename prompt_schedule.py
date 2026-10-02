import re

from comfy_api.latest import io


class H3LoopPromptSchedule(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LoopPromptSchedule",
            display_name="H3 循环分段提示词",
            category="model/conditioning/minimax",
            description="共用描述与分段剧情组合。独占一行的 --- 分隔各段，第1段用于首段，后续段用于续接循环。",
            inputs=[
                io.String.Input("common", display_name="全片固定描述", multiline=True, default=""),
                io.String.Input("segments", display_name="各段剧情（用 --- 分隔）", multiline=True, default="",
                                tooltip="第1块是首段，之后每块对应一次续接。续接次数为N时至少填写N+1块。"),
                io.Int.Input("continuations", display_name="续接次数", default=1, min=0, force_input=True),
            ],
            outputs=[
                io.String.Output(display_name="首段提示词"),
                io.String.Output(display_name="续段提示词列表", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, common, segments, continuations=1):
        sections = [part.strip() for part in re.split(r"(?m)^[ \t]*---[ \t]*\r?$", segments)]
        required = continuations + 1
        if len(sections) < required:
            raise ValueError(f"续接 {continuations} 次需要至少 {required} 段提示词，目前只有 {len(sections)} 段；请用独占一行的 --- 补齐。")
        prompts = []
        for index, section in enumerate(sections[:required], 1):
            if not section:
                raise ValueError(f"第 {index} 段提示词为空，请补写该段剧情。")
            prompts.append("\n\n".join(part for part in (common.strip(), section) if part))
        return io.NodeOutput(prompts[0], prompts[1:])
