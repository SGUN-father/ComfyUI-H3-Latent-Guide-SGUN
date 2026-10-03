# H3-Latent-Guide-SGUN

制作：神棍（SGUN）。主节点：`H3 长视频：分段生成 + 合并 by神棍`。

开发交接、代码结构、参考项目和独立审查重点见同目录的 `开发说明.md`。

一个节点完成 MiniMax H3 Ref2VA 分段采样、原生音视频 LATENT 续接、解码和即时保存。模型、CLIP、视频/音频 VAE、加速、图片和提示词编辑留在外面。每段完成即保存 MP4，可选保存完整 LATENT；合并默认关闭，需要整片时使用外部官方 Save Video。

## 示例工作流

- [普通调度器版：20 秒战斗](examples/H3_长视频_一体节点_20秒战斗.json)：4 段 × 5 秒，使用节点内置调度器。
- [自定义 Sigmas 版](examples/H3_长视频_一体节点_自定义Sigmas.json)：通过 Manual Sigmas 输入采样序列。

下载 JSON 后拖入 ComfyUI，或使用「工作流 → 打开」导入。两个模板都包含参数说明 Note。

配套参考图：[红发少年](examples/red_superboy_on_city_roof.png)、[机械巨兽](examples/mecha_dragon_lightning.png)。将两张图片放入 ComfyUI 的 `input` 目录并在 Load Image 节点选择，也可以替换成自己的图片并修改提示词。

模板保存的是作者本机的模型文件名；导入后按自己的安装情况重新选择 H3 模型、CLIP、视频/音频 VAE 和 Turbo LoRA。注意力后端与稀疏注意力节点来自新版 ComfyUI；若当前环境不支持所选加速后端，将 MODEL 从 LoRA 节点直接连接到 H3 长视频节点即可。合并视频和保存每段 LATENT 默认关闭，需要时在主节点开启。

## 安装与连接

将本目录放入 ComfyUI `custom_nodes`，重启后导入新版配套工作流。不要同时安装旧名和新名两个副本。包来源标识为 `H3-Latent-Guide-SGUN`。使用官方 H3 Ref2VA 权重及其 CLIP、两个 VAE。无需新增依赖，不调用外部命令。

依赖近期 ComfyUI 的官方 H3、V3 API、原生视频接口及自带 PyAV/safetensors；本机验证版本为0.38.0。代码不包含 Windows 专属路径或命令，Linux 尚未实机验证。外部加速节点仍有各自环境要求。

必接 `model`、`clip`、视频 `vae`、`audio_vae`、`prompts`。提示词使用配套 `H3 长视频提示词`：固定描述加入每一段，各段剧情用独占一行的 `---` 分隔。`start_segment` 从1开始；提示词不足默认在采样前报错，`repeat_last=true` 可复用最后一段。

- `segments` 是本次新生成总段数，1→1段、2→2段，默认4。上一段 LATENT 不计数，也不加入本次导出。
- 尺寸和时长沿用官方：24fps、17k+5帧网格、尺寸对齐32。5.0/0.4/16:9 对应864×480；四段为124+119+119+119帧，约20.04秒。首段完整保留，续段自动裁掉重叠和同步音频。音轨按导出帧数截齐，H3 音频 token 取整造成的尾部不足会补零。
- `seed_mode` 使用官方 fixed/increment/decrement/randomize。首段使用面板种子；后续按模式变化，递增/递减在范围边界停止。`control_after_generate` 控制下一次任务的起始种子。多段 `randomize` 每次排队重新随机后续段，即使面板首段种子固定；固定/递增/递减及单段模式仍可复用缓存。段数或种子模式通过连线提供且无法提前判定时，会保守重新执行。
- `sigmas` 可接 Manual Sigmas，接入后直接用于所有片段，覆盖内部 steps/scheduler/denoise；采样器仍生效。未连接则使用面板调度。
- `ref_images`、`ref_videos`、`ref_video_audios`、`ref_audios` 使用官方可扩展接口，各段共用素材。参考视频接24fps IMAGE批次（至少5帧）；视频音轨按相同编号配对。标签编号沿用官方 `<Picture j>`、`<Video j>`、`<Audio j>`。
- 模型和加速在外部调节；内部使用 BasicGuider，无负向 CFG。

## 即时保存与合并

| 参数 | 默认 | 行为 |
| --- | --- | --- |
| 合并视频 `merge_video` | false | 关闭时自动跳过合并输出下游；开启后提供整片 VIDEO，由外部 Save Video 保存。 |
| 保存每段 LATENT `save_latents` | false | 开启后逐段保存完整原生联合音视频 `.h3latent`，不累积所有完整张量。 |
| 分段保存前缀 `filename_prefix` | video/H3-SGUN/H3 | 在 ComfyUI output 内创建独立编号目录，按段号配对保存文件。 |

无论合并开关如何，视频均**每段完成立即写入磁盘**。主节点是输出节点；配套工作流已移除重复保存片段的外部 Save Video，保留一个可选合并保存节点。停止后本地已完成文件保留，未完成的临时写入文件清理。

每次任务目录示例：

```text
output/video/H3-SGUN/H3_00001/
    segment_0001.mp4
    segment_0001.h3latent    # 开启 LATENT 保存才生成
    segment_0002.mp4
    segment_0002.h3latent
    segments.jsonl         # 逐段追加实际种子、帧数、路径
```

视频使用官方默认 H.264/AAC MP4，当前 H.264 默认 CRF 为18，不提供主节点覆盖。合并使用外部 Save Video 的 auto/MP4、codec auto，可按官方规则流式拼接视频包及音轨；其他编码/CRF可能触发转码。

五个输出依次为合并 VIDEO、最后一段完整音视频 LATENT、生成信息 JSON、分段 VIDEO 列表、LATENT 文件路径列表。分段 VIDEO 指向已经保存的磁盘文件，无需重复保存。合并关闭时只阻断合并输出，其他输出正常。主节点仅预览最后一个片段，全部路径仍保留在生成信息/列表输出中，避免同时显示大量预览。LATENT 保存关闭时路径列表为空，内部续接及最后一段 LATENT 输出仍正常。

生成信息包含实际尺寸/时长/帧数、实际种子、`continued_from_latent`、开关状态、目录、视频/LATENT路径、JSONL清单和CPU尾部缓存字节数。文件号从本次第1段重新编号，提示词起始段号只选择文本。已完成段的实际种子也写入JSONL，停止后可查找。

## LATENT 续接与重抽

`previous_latent`（上一段 LATENT）替换原 `previous_video`。接上一生成节点的“最后一段 latent”，或配套 `H3 读取音视频 LATENT by神棍`。必须包含完整 H3 视频和音频流，且与新段分辨率一致。直接复制原生尾部上下文，不经过视频压缩或 VAE 往返。

跨任务重抽需提前开启“保存每段 LATENT”。重抽第3段：读取第2段 `.h3latent` → 上一段 LATENT；提示词 `start_segment=3`；生成段数=1；保持分辨率，调整种子。比较相同噪声时手动使用原第3段实际种子。输入段不出现在新输出中；后续动作需以重抽的新段继续生成。

读取节点直接在“已保存的 LATENT”下拉列表里选择文件，无需复制路径。自动扫描 ComfyUI `output`、`input`、`temp` 及子目录中的 `.h3latent`，按修改时间排序，最新文件在最前面。生成新片段后点击列表下方的刷新按钮；刷新会选择最新文件，重抽其他段时再选择对应段号。未保存 LATENT 时，列表会提示先开启主节点的“保存每段 LATENT”。

高级选项中的“手动路径（可留空）”保留旧工作流兼容：通常留空；已有路径或路径连线优先于下拉选择。路径仍使用原生 `[output]`/`[input]`/`[temp]` 标记，不带标记默认 input，限定在这些目录内。

`.h3latent` 是 safetensors 格式，分别保存视频/音频张量，保留 dtype、数值及音频取整偏移；使用本插件读取。普通 Save/Load Latent 不支持这类联合 NestedTensor。文件内容变化会更新读取节点缓存指纹。

不接上一段 LATENT 且生成段数=1，可正常生成完整首段。未提前保存 LATENT 时，不能从 MP4 恢复原采样数值；可在同一图直接传最后一段 LATENT，或重新生成需要的段。升级后导入新版一体工作流，旧 VIDEO 接口需改接 LATENT。

## 动态掩码

`dynamic_mask`（动态掩码）默认false。只有接入上一段LATENT或内部续段才生效；全新首段保持普通生成流程。关闭时仍使用原来的H3 keyframe引导。

开启后，把上一段尾部复制到本段将被裁掉的前缀，在采样中按实际sigma调整视频软掩码：远离接缝的上下文允许更多重绘，紧邻接缝的位置保持锁定；新生成区域正常采样，音频上下文保持锁定。采样掩码与H3内部token掩码使用同一套官方映射，避免两者不一致。内部调度和外接SIGMAS均不改写；不增加第二遍采样，不修改保存的上一段LATENT或输入MODEL。

这种机制用于尝试缓解续接漂移、接缝及上下文锁定带来的问题，不能恢复已经丢失的细节，也不能保证越续越清晰。参考项目作者的测试也报告过运动更稳定、清晰度稍降的取舍，参见 [Easy Media实现与作者说明](https://github.com/yolain/ComfyUI-Easy-Media/blob/main/modules/motion_context/drift_control_av.py)。本节点采用独立实现，未证明同样的画质收益；建议用相同模型、参考图、提示词和实际种子分别生成开/关两组，比较后半段。

需要ComfyUI具备原生H3音视频软掩码和ModelPatcher钩子支持；相关官方支持见 [ComfyUI PR #15375](https://github.com/Comfy-Org/ComfyUI/pull/15375)。版本缺少接口或MODEL已经装有其他动态掩码时，会在采样前明确报错，可更新或关闭本开关。加速模型通过克隆保留原有配置，具体加速器仍需实测。每段有少量工作掩码开销，采样完成或异常时释放活动掩码，不跨段累积。

生成信息记录开关值和逐段`continuation_methods`；JSONL清单逐段记录`continuation_method`。`none`为无前段首段，`guide`为普通续接，`dynamic_mask`为本次动态掩码续接。

## 资源与平台

下一段开始前只保留独立复制的CPU音视频尾部，不累积完整中间 LATENT、原始画面或压缩视频缓冲。默认22帧上下文保留39帧完整AV网格，容纳至少一秒音频；最后一段完整 LATENT 作为输出保留。磁盘路径及预览记录仍随段数增长，系统文件缓存也可能增长。

关闭合并省去整片保存、编码/转码处理和整片文件；两种模式均使用磁盘片段。保存 LATENT 需要本段CPU数据写入和额外磁盘空间，不输出整片张量列表。显存主要仍由模型、单段采样和解码决定；外部节点若解码整片为 IMAGE，仍可能占用大量内存。

先运行1–4段，并发1，观察显存、系统内存和可用磁盘再增加段数。资源不足先降低百万像素、每段时长和参考规模。开关不是无限生成或低配置安全保证，没有验证各类设备的统一安全段数。

H3 解码调用官方 VAE Decode，内部原生分块当前使用空间256像素、最低空间重叠64像素、时间切片基准17帧；通用分块参数不控制H3后端，已移除这些控件。参见 [官方配置](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/Ref2VA/video_vae/config.json) 与 [ComfyUI 实现](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/vae.py)。固定图片和描述有助一致性，原生 LATENT 传递不能保证无限长链不劣化。

RunningHub/Linux 尚未实机测试。节点使用原生 ComfyUI，不依赖自定义网页或服务；平台需安装本插件和所需官方版本。云端生成目录不等于跨任务存储，需平台支持保存/下载/上传 `.h3latent`；任务失败时是否保留、展示已保存文件也由平台决定。

## 保留的独立引导与循环工具

`MiniMax H3 潜空间续接引导` 接下一段 positive、空联合 LATENT、上一段完整采样 LATENT、context_frames，输出 positive 和 trim_frames。下一段仍采样新的空 LATENT；正常解码后删掉前 trim_frames 帧和 trim_frames/24 秒音频再拼接。两段须同分辨率、batch=1。对齐后的上下文必须小于本段总帧数，否则提前报错，避免裁剪后没有新画面。

`H3 循环分段提示词` 保留旧循环用法：首段在循环外，Start Loop的initial_iteration_value接首段 LATENT，current_iteration_value接引导previous_latent，续段采样结果接End Loop的next_iteration_value。旧工具的循环次数仍按续接次数计算，不改变新主节点的总段数语义。

## 验证

16项原生引导、31项集成和5项动态掩码测试通过，共52项；另有11项不依赖ComfyUI的标准库边界测试通过。集成检查包括四种种子/边界、提示词顺序与重抽段选择、参考传递、自定义sigmas、四段实际PyAV保存、合并及视频包一致性、音轨/元数据、跨段大张量释放、合并阻断SaveVideo、主节点原生执行、三种dtype的LATENT逐值恢复、路径限制、中止后保留已完成文件和临时文件清理。动态掩码测试覆盖原生H3掩码映射与inpaint路径、音视频上下文对齐、输入张量不变、实际SIGMAS不变、首段/续段切换、版本/钩子冲突以及成功/失败时工作掩码释放。新增检查覆盖递归文件列表、最新文件排序、刷新接口、下拉读取及旧路径优先级；音轨补零/裁剪、真实H3音频长度取整、空轨、dtype/device保留；官方缓存键下重复排队、确定模式复用、连线模式失效；上下文全覆盖提前拒绝。两个配套工作流及动态掩码/合并/LATENT保存开关组合、单段LATENT续接通过原生校验。

这次使用小尺寸CPU模拟模型预测，实际调用原生H3掩码/inpaint、视频编码、保存/读取和执行器；未重新运行H3模型、测量长片资源峰值或验证RunningHub。此前模型生成记录不代表本次版本资源峰值；动态掩码对真实长链画质的影响尚未实测。

可用 ComfyUI 的 Python 运行 `tests/test_latent_guide.py`、`tests/test_long_video.py`、`tests/test_dynamic_mask.py`，无需新测试依赖。实现基于官方 H3 keyframe 接口，参考 [H3 Continuation](https://github.com/ttulttul/ComfyUI-Minimax-H3-Continuation)、[H3 Motion Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)，不依赖这两个第三方节点包。

标准库边界测试可执行 `python -B -X utf8 tests/test_boundary_regressions.py`。2026-10-03 合入已审查的音轨长度、独立引导上下文和随机模式缓存修复，保留 LATENT 下拉加载与示例工作流。
