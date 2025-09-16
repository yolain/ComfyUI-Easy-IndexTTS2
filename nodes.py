import os
import random
import re
import shutil
import folder_paths
import yaml
import numpy as np
import torch
import gc
import json
import hashlib
import comfy.model_management as mm
import comfy.utils
from typing import Tuple, Any
from comfy_api.latest import ComfyExtension, io
from .indextts2.infer import IndexTTS2Engine
from .indextts2.model_loader import IndexTTS2Loader

from server import PromptServer

# 初始化模型加载器和引擎
loader = IndexTTS2Loader()
engine = IndexTTS2Engine(loader)
fingerprint = 1

# 定义自定义类型
TYPE_Audios = io.Custom(io_type="AUDIOS")
TYPE_IndexTTSModel = io.Custom(io_type="EASY_INDEXTTS_MODEL")
TYPE_Emotions = io.Custom(io_type="EASY_INDEXTTS_EMOTIONS")
# 情感模式
class EMOTION_MODE:
    SAME_AS_REF = 0
    EMO_AUDIO = 1
    EMO_VECTOR = 2
    EMO_TEXT = 3

# 处理音频输入
def process_audio_input(audio: io.Audio) -> Tuple[np.ndarray, int]:
    if isinstance(audio, dict) and "waveform" in audio and "sample_rate" in audio:
        wave = audio["waveform"]
        sr = int(audio["sample_rate"])
        if isinstance(wave, torch.Tensor):
            if wave.dim() == 3:
                wave = wave[0, 0].detach().cpu().numpy()
            elif wave.dim() == 1:
                wave = wave.detach().cpu().numpy()
            else:
                wave = wave.flatten().detach().cpu().numpy()
        elif isinstance(wave, np.ndarray):
            if wave.ndim == 3:
                wave = wave[0, 0]
            elif wave.ndim == 2:
                wave = wave[0]
        return wave.astype(np.float32), sr
    elif isinstance(audio, tuple) and len(audio) == 2:
        wave, sr = audio
        if isinstance(wave, torch.Tensor):
            wave = wave.detach().cpu().numpy()
        return wave.astype(np.float32), int(sr)
    else:
        raise ValueError("AUDIO input must be ComfyUI dict or (wave, sr)")

# 下载并加载IndexTTS2模型节点
class DownloadAndLoadIndexTTSModel(io.ComfyNode):
    pass

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy downloadIndexTTSAndLoadModel",
            display_name="IndexTTS Model Loader",
            category="EasyUse/IndexTTS2",
            inputs=[
                io.Combo.Input("model", options=["IndexTTS-2",]),
                io.Combo.Input("download_from", options=["huggingface", "modelscope"], default="huggingface"),
                io.AnyType.Input("start", optional=True, tooltip="Start to load or download models"),
            ],
            outputs=[
                TYPE_IndexTTSModel.Output(display_name="indextts_model"),
            ],
            hidden=[io.Hidden.unique_id]
        )

    @classmethod
    def _download_from_huggingface(cls, model_id: str, local_path: str, subfolder: str = None, filename: str = None):
        """从Hugging Face下载模型"""
        try:
            from huggingface_hub import snapshot_download, hf_hub_download
            if filename:
                # 单文件下载
                file_path = subfolder + "/" + filename if subfolder else filename
                downloaded_file = hf_hub_download(
                    repo_id=model_id,
                    filename=file_path,
                    local_dir=local_path,
                    local_dir_use_symlinks=False
                )
                return downloaded_file
            else:
                # 全仓库下载
                if subfolder:
                    snapshot_download(repo_id=model_id, local_dir=local_path, allow_patterns=[f"{subfolder}/*"])
                else:
                    snapshot_download(repo_id=model_id, local_dir=local_path)
        except ImportError:
            raise Exception("huggingface_hub not installed. Please run: pip install huggingface_hub")

    @classmethod
    def _download_from_modelscope(cls, model_id: str, local_path: str, subfolder: str = None, filename: str = None):
        """从ModelScope下载模型"""
        cache_dir = os.path.join(local_path, "cache")
        try:
            from modelscope import snapshot_download
            if filename:
                # 单文件下载 - ModelScope目前没有直接的单文件下载API，使用snapshot_download + allow_patterns
                file_pattern = subfolder + "/" + filename if subfolder else filename
                downloaded_path = snapshot_download(
                    model_id,
                    cache_dir=cache_dir,
                    allow_patterns=[file_pattern]
                )
                # 返回下载的文件的完整路径
                downloaded_file = os.path.join(downloaded_path, file_pattern)
                return downloaded_file
            else:
                # 全仓库下载
                if subfolder:
                    snapshot_download(model_id, local_dir=local_path, allow_patterns=[f"{subfolder}/*"])
                else:
                    snapshot_download(model_id, local_dir=local_path,)
        except ImportError:
            raise Exception("modelscope not installed. Please run: pip install modelscope")

    @classmethod
    async def fingerprint_inputs(cls, model: str, download_from: str, start=None) -> str:
        global fingerprint
        # 构建基于输入参数的指纹
        base_fingerprint = f"{model}_{download_from}"
        return f"{base_fingerprint}_{str(fingerprint)}"

    @classmethod
    def execute(cls, model: str, download_from: str, start: io.AnyType = None) -> io.NodeOutput:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        model_root_dir = os.path.join(folder_paths.models_dir, model)

        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)

            # 获取模型配置
            model_config = config.get('models', {}).get(model, [])
            if not model_config:
                raise ValueError(f"Model '{model}' not found in config.yaml")

            for item in model_config:
                type = item.get('type', 'multi')
                model_name = item.get('name')
                model_id = item.get(download_from)
                dir_subfolder = item.get('dir', '')
                dir_name = item.get('dir_name', None)
                file = item.get('file', None)

                # 首先判断模型根目录是否存在
                if model_name == "IndexTTS-2":
                    required = [
                        "bpe.model",
                        "config.yaml",
                        "feat1.pt",
                        "feat2.pt",
                        "gpt.pth",
                        "s2mel.pth",
                        "wav2vec2bert_stats.pt",
                        "qwen0.6bemo4-merge"
                    ]
                    missing = [f for f in required if not os.path.exists(os.path.join(model_root_dir, f))]
                    if missing:
                        # 根据来源下载模型
                        if download_from == "huggingface":
                            cls._download_from_huggingface(
                                model_id=model_id,
                                local_path=model_root_dir,
                                subfolder=dir_subfolder if dir_subfolder else None
                            )
                        elif download_from == "modelscope":
                            cls._download_from_modelscope(
                                model_id=model_id,
                                local_path=model_root_dir,
                                subfolder=dir_subfolder if dir_subfolder else None
                            )
                else:
                    if type == 'single':
                        # 单文件下载到根路径
                        if not file:
                            print(f"Warning: No file specified for single type model {model_name}")
                            continue
                        
                        target_file_path = os.path.join(model_root_dir, file)
                        
                        # 检查文件是否已经存在
                        if not os.path.exists(target_file_path):
                            print(f"Downloading single file {file} from {download_from}...")
                            
                            # 确保目标目录存在
                            os.makedirs(model_root_dir, exist_ok=True)
                            
                            try:
                                # 根据来源下载单个文件
                                if download_from == "huggingface":
                                    downloaded_file = cls._download_from_huggingface(
                                        model_id=model_id,
                                        local_path=model_root_dir,
                                        subfolder=dir_subfolder if dir_subfolder else None,
                                        filename=file
                                    )
                                    if downloaded_file and downloaded_file != target_file_path:
                                        # 如果下载的文件路径不是我们期望的，需要移动
                                        shutil.move(downloaded_file, target_file_path)
                                elif download_from == "modelscope":
                                    downloaded_file = cls._download_from_modelscope(
                                        model_id=model_id,
                                        local_path=model_root_dir,
                                        subfolder=dir_subfolder if dir_subfolder else None,
                                        filename=file
                                    )
                                    if downloaded_file and downloaded_file != target_file_path:
                                        # 如果下载的文件路径不是我们期望的，需要移动
                                        shutil.move(downloaded_file, target_file_path)
                                
                                print(f"Downloaded {file} to {target_file_path}")
                                
                            except Exception as e:
                                # 如果下载失败，删除可能存在的不完整文件
                                if os.path.exists(target_file_path):
                                    os.remove(target_file_path)
                                raise e
                        else:
                            print(f"Single file {file} already exists at {target_file_path}")
                    else:
                        # 多文件下载，需要移动到指定目录
                        if not dir_name:
                            print(f"Warning: No dirname specified for multi-file model {model_name}")
                            continue
                        
                        # 最终目标路径
                        final_target_path = os.path.join(model_root_dir, dir_name)
                        
                        # 检查目标目录是否已经存在
                        if not os.path.exists(final_target_path):
                            print(f"Downloading {model_name} from {download_from}...")
                            cache_path = os.path.join(model_root_dir, model_id)
                            try:
                                # 根据来源下载模型
                                if download_from == "huggingface":
                                    cls._download_from_huggingface(
                                        model_id=model_id,
                                        local_path=cache_path,
                                        subfolder=dir_subfolder if dir_subfolder else None
                                    )
                                elif download_from == "modelscope":
                                    cls._download_from_modelscope(
                                        model_id=model_id,
                                        local_path=cache_path,
                                        subfolder=dir_subfolder if dir_subfolder else None
                                    )

                                temp_download_dir = os.path.join(model_root_dir, model_id, dir_name) if dir_subfolder else os.path.join(model_root_dir, model_id)
                                shutil.move(temp_download_dir, final_target_path)
                                print(f"Downloaded and moved {model_name} to {final_target_path}")

                            except Exception as e:
                                raise e
                        else:
                            print(f"Model {model_name} already exists at {final_target_path}")

            # 加载模型
            engine.tts = engine.loader.get_tts()
            # PromptServer.instance.send_progress_text(
            #     f"{model} loaded successfully.",
            #     cls.hidden.unique_id
            # )
            return io.NodeOutput(engine)
        except yaml.YAMLError:
            raise Exception("Error parsing config.yaml. Please check the YAML syntax.")
        except Exception as e:
            raise Exception(f"Download failed: {str(e)}")
# 语音节点
class indexTTS2Generate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy indexTTSGenerate",
            display_name="IndexTTS Generate",
            category="EasyUse/IndexTTS2",
            inputs=[
                TYPE_IndexTTSModel.Input("indextts_model"),
                io.Audio.Input("reference_audio",  optional=True, tooltip="Reference audio for voice cloning (Only the one voice is supported)"),
                TYPE_Audios.Input("reference_audios", optional=True, tooltip="(Optional) Reference audios for voice cloning (Multiple voices are supported)"),
                TYPE_Emotions.Input("emotions", optional=True, tooltip="(Optional) voice emotions"),
                io.String.Input("text", multiline=True, placeholder="", tooltip="Text to synthesize. Supports pause format: -2s- (2 seconds pause), -0.5s- (0.5 seconds pause)"),
                io.Boolean.Input("unload_model", label_on="on", label_off="off", default=False, tooltip="Unload model from VRAM after synthesis"),
                io.Boolean.Input("do_sample", label_on="on", label_off="off", default=True),
                io.Float.Input("temperature", default=0.8, min=0.1, max=2.0, step=0.05),
                io.Float.Input("top_p", default=0.9, min=0.0, max=1.0, step=0.01),
                io.Int.Input("top_k", default=30, min=0, max=100, step=1),
                io.Int.Input("num_beams", default=3, min=1, max=10, step=1),
                io.Float.Input("repetition_penalty", default=10.0, min=1.0, max=10.0, step=0.1),
                io.Float.Input("length_penalty", default=0.0, min=-2.0, max=2.0, step=0.1),
                io.Int.Input("max_mel_tokens", default=1815, min=50, max=1815, step=5),
                io.Int.Input("max_tokens_per_sentence", default=120, min=0, max=600, step=5),
                io.Int.Input("seed", default=0, min=0, max=2 ** 32 - 1),
            ],
            outputs=[
                io.Audio.Output(display_name="audio"),
                io.Int.Output(display_name="seed"),
                io.String.Output(display_name="subtitle"),
            ],
            hidden=[io.Hidden.unique_id]
        )

    @classmethod
    def execute(cls, indextts_model, text: str, unload_model:bool, do_sample: bool, temperature: float, top_p: float, top_k: int, num_beams: int, repetition_penalty: float, length_penalty: float, max_mel_tokens: int, max_tokens_per_sentence: int, seed: int, reference_audio=None, reference_audios=None, emotions=None, unique_id=None) -> io.NodeOutput:
        global fingerprint
        # 音频预处理
        if emotions is None and reference_audio is None and reference_audios is None:
            raise ValueError("Please provide either emotions or reference_audio for voice cloning.")
        
        if emotions is not None and len(emotions) > 0:
            # 生成多音色情感语音
            voice_map = {}
            voice_order = []
            
            # 构建音色映射表，为没有名字的音色分配默认名字
            for idx, emotion_data in enumerate(emotions):
                vc_name = emotion_data.get("voice_name")
                if vc_name is None or vc_name == "":
                    vc_name = f"s{idx + 1}"
                voice_map[vc_name] = emotion_data
                if vc_name not in voice_order:
                    voice_order.append(vc_name)
            
            # 解析文本格式 [voice_name] 对话内容
            lines = text.split('\n')
            segments = []
            current_char = None
            current_text = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 检查是否是停顿标识 -Xs- (X可以是整数或浮点数)
                pause_match = re.match(r'^-(\d+(?:\.\d+)?)s-$', line)
                if pause_match:
                    # 保存之前的片段
                    if current_char and current_text.strip():
                        segments.append((current_char, current_text.strip()))
                        current_char = None
                        current_text = ""
                    
                    # 添加停顿片段
                    pause_duration = float(pause_match.group(1))
                    segments.append(("__PAUSE__", pause_duration))
                    continue
                    
                # 检查是否是音色标识行 [voice_name]
                if line.startswith('[') and ']' in line:
                    # 保存之前的片段
                    if current_char and current_text.strip():
                        segments.append((current_char, current_text.strip()))
                    
                    # 提取角色名
                    end_bracket = line.find(']')
                    current_char = line[1:end_bracket].strip()
                    current_text = line[end_bracket + 1:].strip()
                else:
                    current_text += " " + line if current_text else line
            
            # 保存最后一个片段
            if current_char and current_text.strip():
                segments.append((current_char, current_text.strip()))
            
            # 如果没有找到音色标识，使用默认第一个音色
            if not segments and text.strip():
                default_char = voice_order[0] if voice_order else "s1"
                segments = [(default_char, text.strip())]
            
            # 生成每个片段的音频
            all_waves = []
            all_subtitles = []
            current_time = 0.0
            current_sr = None  # 用于存储当前采样率
            # 缓存已处理的角色参考音频，避免重复处理
            processed_ref_audios = {}
            processed_emo_audios = {}
            
            # 初始化进度条
            pbar = comfy.utils.ProgressBar(len(segments))
            
            for segment_idx, segment in enumerate(segments):
                if len(segment) == 2 and segment[0] == "__PAUSE__":
                    # 处理停顿
                    pause_duration = segment[1]
                    # 如果还没有采样率信息，使用默认值22050
                    sample_rate = current_sr if current_sr is not None else 22050
                    silence_samples = int(pause_duration * sample_rate)
                    silence_wave = np.zeros(silence_samples, dtype=np.float32)
                    all_waves.append(silence_wave)
                    
                    # 添加停顿字幕
                    all_subtitles.append({
                        "id": "pause",
                        "字幕": f"[停顿 {pause_duration}秒]",
                        "start": round(current_time, 2),
                        "end": round(current_time + pause_duration, 2)
                    })
                    
                    current_time += pause_duration
                    continue
                
                vc_name, segment_text = segment
                # 查找对应的音色情感配置
                if vc_name not in voice_map:
                    # 如果音色名不存在，使用第一个可用音色
                    vc_name = voice_order[0] if voice_order else "s1"
                
                emotion_data = voice_map[vc_name]
                # 检查是否已经处理过该音色的参考音频
                if vc_name not in processed_ref_audios:
                    processed_ref_audios[vc_name] = process_audio_input(emotion_data["reference_audio"])
                ref_audio = processed_ref_audios[vc_name]
                
                emo_mode = emotion_data.get("emo_mode", EMOTION_MODE.SAME_AS_REF)
                
                # 根据情感模式设置参数
                emo_text_param = None
                emo_ref_audio_param = None
                emo_vector_param = None
                use_qwen = False
                emo_weight_param = 0.8
                use_random_param = emotion_data.get("use_random", False)
                
                if emo_mode == EMOTION_MODE.EMO_TEXT:
                    emo_text_param = emotion_data.get("description", "")
                    use_qwen = True
                elif emo_mode == EMOTION_MODE.EMO_AUDIO:
                    # 检查是否已经处理过该音色的情感参考音频
                    emo_audio_key = f"{vc_name}_emo"
                    if emo_audio_key not in processed_emo_audios and "emo_ref_audio" in emotion_data:
                        processed_emo_audios[emo_audio_key] = emotion_data.get("emo_ref_audio")
                    emo_ref_audio_param = processed_emo_audios.get(emo_audio_key)
                    emo_weight_param = emotion_data.get("emo_weight", 0.8)
                elif emo_mode == EMOTION_MODE.EMO_VECTOR:
                    emo_vector_param = emotion_data.get("emo_vector")

                print(f"Generating for voice: {vc_name}, Text: {segment_text[:30]}..., EmoMode: {emo_mode}, emo_text_param: {emo_text_param}")
                # 生成单个片段的音频
                sr, wave, sub = indextts_model.generate(
                    text=segment_text, 
                    reference_audio=ref_audio, 
                    mode="Auto",
                    do_sample=do_sample, 
                    temperature=temperature, 
                    top_p=top_p, 
                    top_k=top_k, 
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty, 
                    length_penalty=length_penalty,
                    max_mel_tokens=max_mel_tokens, 
                    max_tokens_per_sentence=max_tokens_per_sentence,
                    emo_text=emo_text_param, 
                    emo_ref_audio=emo_ref_audio_param, 
                    emo_vector=emo_vector_param, 
                    emo_weight=emo_weight_param,
                    seed=seed, 
                    return_subtitles=True, 
                    use_random=use_random_param,
                    use_qwen=use_qwen
                )
                
                # 更新当前采样率
                if current_sr is None:
                    current_sr = sr
                
                all_waves.append(wave)
                
                # 计算片段时长并更新字幕时间
                segment_duration = len(wave) / float(sr)
                if sub:
                    try:
                        sub_data = json.loads(sub)
                        for item in sub_data:
                            item["id"] = vc_name
                            item["start"] = round(current_time + item.get("start", 0), 2)
                            item["end"] = round(current_time + item.get("end", segment_duration), 2)
                        all_subtitles.extend(sub_data)
                    except:
                        # 如果解析失败，创建简单字幕
                        all_subtitles.append({
                            "id": vc_name,
                            "字幕": segment_text,
                            "start": round(current_time, 2),
                            "end": round(current_time + segment_duration, 2)
                        })
                else:
                    all_subtitles.append({
                        "id": vc_name,
                        "字幕": segment_text,
                        "start": round(current_time, 2),
                        "end": round(current_time + segment_duration, 2)
                    })
                
                current_time += segment_duration
                
                # 更新进度条
                pbar.update(1)
            
            # 连接所有音频片段
            final_wave = np.concatenate(all_waves) if all_waves else np.array([])
            wave_t = torch.tensor(final_wave, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            audio = {"waveform": wave_t, "sample_rate": int(sr)}
            
            # 生成最终字幕
            final_subtitle = json.dumps(all_subtitles, ensure_ascii=False) if all_subtitles else ""

            # 卸载模型
            if unload_model:
                indextts_model.unload_model()
                fingerprint = random.randrange(100000, 999999)

            return io.NodeOutput(audio, seed, final_subtitle)
        elif reference_audios is not None and len(reference_audios) > 0:
            # 使用reference_audios生成多角色语音
            # 解析文本格式 [s1: emotion_data] 对话内容
            lines = text.split('\n')
            segments = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 检查是否是停顿标识 -Xs- (X可以是整数或浮点数)
                pause_match = re.match(r'^-(\d+(?:\.\d+)?)s-$', line)
                if pause_match:
                    # 添加停顿片段
                    pause_duration = float(pause_match.group(1))
                    segments.append(("__PAUSE__", pause_duration))
                    continue
                
                # 检查是否符合 [sx: emotion_data] 或 [sx] 格式
                if line.startswith('[') and ']' in line:
                    # 查找格式结束位置
                    end_pos = line.find(']')
                    if end_pos == -1:
                        raise ValueError(f"格式不正确: {line}")
                    
                    # 提取音色和情感数据
                    format_part = line[1:end_pos].strip()
                    dialog_text = line[end_pos + 1:].strip()
                    
                    # 分离音色名和情感数据（可选）
                    if ':' in format_part:
                        vc_name, emotion_data = format_part.split(':', 1)
                        vc_name = vc_name.strip()
                        emotion_data = emotion_data.strip()
                    else:
                        vc_name = format_part.strip()
                        emotion_data = ""
                    
                    # 验证音色名格式 (s1, s2, s3...)
                    if not vc_name.startswith('s') or not vc_name[1:].isdigit():
                        raise ValueError(f"格式不正确，音色名必须为s1, s2格式: {vc_name}")
                    
                    char_index = int(vc_name[1:]) - 1
                    if char_index < 0 or char_index >= len(reference_audios):
                        raise ValueError(f"音色索引超出范围: {vc_name}, 可用音频数量: {len(reference_audios)}")
                    
                    # 判断情感数据类型
                    emo_mode = EMOTION_MODE.SAME_AS_REF  # 默认使用参考音频的情感
                    emo_vector = None
                    emo_text = None
                    
                    # 如果有情感数据，进行解析
                    if emotion_data:
                        # 检查是否为8个数字（向量格式）
                        if ',' in emotion_data:
                            try:
                                values = [float(x.strip()) for x in emotion_data.split(',')]
                                if len(values) == 8:
                                    emo_mode = EMOTION_MODE.EMO_VECTOR
                                    emo_vector = values
                                else:
                                    raise ValueError(f"情感向量必须包含8个数值: {emotion_data}")
                            except ValueError as e:
                                if "情感向量必须包含8个数值" in str(e):
                                    raise e
                                # 如果不是数字，当作文本处理
                                emo_mode = EMOTION_MODE.EMO_TEXT
                                emo_text = emotion_data
                        else:
                            # 尝试解析为单个数字
                            try:
                                float(emotion_data)
                                raise ValueError(f"单个数值无效，请提供8个逗号分隔的数值或文本描述: {emotion_data}")
                            except ValueError:
                                # 当作文本处理
                                emo_mode = EMOTION_MODE.EMO_TEXT
                                emo_text = emotion_data
                    
                    segments.append((char_index, vc_name, dialog_text, emo_mode, emo_vector, emo_text))
                else:
                    raise ValueError(f"格式不正确，必须使用 [sx: emotion_data] 或 [sx] 对话内容 格式或 -Xs- 停顿格式: {line}")
            
            if not segments:
                raise ValueError("未找到有效的对话片段")
            
            # 生成每个片段的音频
            all_waves = []
            all_subtitles = []
            current_time = 0.0
            current_sr = None  # 用于存储当前采样率
            # 缓存已处理的音色参考音频，避免重复处理
            processed_ref_audios = {}
            
            # 初始化进度条
            pbar = comfy.utils.ProgressBar(len(segments))
            
            for segment_idx, segment in enumerate(segments):
                if len(segment) == 2 and segment[0] == "__PAUSE__":
                    # 处理停顿
                    pause_duration = segment[1]
                    # 如果还没有采样率信息，使用默认值22050
                    sample_rate = current_sr if current_sr is not None else 22050
                    silence_samples = int(pause_duration * sample_rate)
                    silence_wave = np.zeros(silence_samples, dtype=np.float32)
                    all_waves.append(silence_wave)
                    
                    # 添加停顿字幕
                    all_subtitles.append({
                        "id": "pause",
                        "字幕": f"[停顿 {pause_duration}秒]",
                        "start": round(current_time, 2),
                        "end": round(current_time + pause_duration, 2)
                    })
                    
                    current_time += pause_duration
                    continue
                
                char_index, vc_name, segment_text, emo_mode, emo_vector, emo_text = segment
                # 检查是否已经处理过该音色的参考音频
                if char_index not in processed_ref_audios:
                    processed_ref_audios[char_index] = process_audio_input(reference_audios[char_index])
                ref_audio = processed_ref_audios[char_index]
                
                # 调试信息：确认音频对应关系
                # print(f"Using voice {vc_name} (index {char_index}) for text: {segment_text[:30]}...")
                # print(f"Reference audio shape: {ref_audio[0].shape if isinstance(ref_audio, tuple) else 'N/A'}, Sample rate: {ref_audio[1] if isinstance(ref_audio, tuple) else 'N/A'}")
                
                # 根据情感模式设置参数
                emo_text_param = None
                emo_ref_audio_param = None
                emo_vector_param = None
                use_qwen = False
                emo_weight_param = 0.8
                use_random_param = False
                
                if emo_mode == EMOTION_MODE.EMO_TEXT:
                    emo_text_param = emo_text
                    use_qwen = True
                elif emo_mode == EMOTION_MODE.EMO_VECTOR:
                    emo_vector_param = emo_vector
                
                # print(f"Generating for voice: {vc_name}, Text: {segment_text[:30]}..., EmoMode: {emo_mode}, emo_text_param: {emo_text_param}")
                # 生成单个片段的音频
                sr, wave, sub = indextts_model.generate(
                    text=segment_text, 
                    reference_audio=ref_audio, 
                    mode="Auto",
                    do_sample=do_sample, 
                    temperature=temperature, 
                    top_p=top_p, 
                    top_k=top_k, 
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty, 
                    length_penalty=length_penalty,
                    max_mel_tokens=max_mel_tokens, 
                    max_tokens_per_sentence=max_tokens_per_sentence,
                    emo_text=emo_text_param, 
                    emo_ref_audio=emo_ref_audio_param, 
                    emo_vector=emo_vector_param, 
                    emo_weight=emo_weight_param,
                    seed=seed, 
                    return_subtitles=True, 
                    use_random=use_random_param,
                    use_qwen=use_qwen
                )
                
                # 更新当前采样率
                if current_sr is None:
                    current_sr = sr
                
                all_waves.append(wave)
                
                # 计算片段时长并更新字幕时间
                segment_duration = len(wave) / float(sr)
                if sub:
                    try:
                        sub_data = json.loads(sub)
                        for item in sub_data:
                            item["id"] = vc_name
                            item["start"] = round(current_time + item.get("start", 0), 2)
                            item["end"] = round(current_time + item.get("end", segment_duration), 2)
                        all_subtitles.extend(sub_data)
                    except:
                        # 如果解析失败，创建简单字幕
                        all_subtitles.append({
                            "id": vc_name,
                            "字幕": segment_text,
                            "start": round(current_time, 2),
                            "end": round(current_time + segment_duration, 2)
                        })
                else:
                    all_subtitles.append({
                        "id": vc_name,
                        "字幕": segment_text,
                        "start": round(current_time, 2),
                        "end": round(current_time + segment_duration, 2)
                    })
                
                current_time += segment_duration
                
                # 更新进度条
                pbar.update(1)
            
            # 连接所有音频片段
            final_wave = np.concatenate(all_waves) if all_waves else np.array([])
            wave_t = torch.tensor(final_wave, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            audio = {"waveform": wave_t, "sample_rate": int(sr)}
            
            # 生成最终字幕
            final_subtitle = json.dumps(all_subtitles, ensure_ascii=False) if all_subtitles else ""

            # 卸载模型
            if unload_model:
                indextts_model.unload_model()
                fingerprint = random.randrange(100000, 999999)

            return io.NodeOutput(audio, seed, final_subtitle)
        else:
            # 生成单人语音
            pbar = comfy.utils.ProgressBar(100)
            ref = process_audio_input(reference_audio)
            pbar.update(10)
            sr, wave, sub = indextts_model.generate(text=text, reference_audio=ref, mode="Auto",
                do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams,
                repetition_penalty=repetition_penalty, length_penalty=length_penalty,
                max_mel_tokens=max_mel_tokens, max_tokens_per_sentence=max_tokens_per_sentence,
                emo_text=None, emo_ref_audio=None, emo_vector=None, emo_weight=0.8,
                seed=seed, return_subtitles=True, use_random=False)
            pbar.update(100)
            wave_t = torch.tensor(wave, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            audio = {"waveform": wave_t, "sample_rate": int(sr)}

            # 卸载模型
            if unload_model:
                indextts_model.unload_model()
                fingerprint = random.randrange(100000, 999999)

            return io.NodeOutput(audio, seed, (sub or ""))

# 简化版语音生成节点
class indexTTSGenerateSimple(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy indexTTSGenerateSimple",
            display_name="IndexTTS Generate Simple",
            category="EasyUse/IndexTTS2",
            inputs=[
                TYPE_IndexTTSModel.Input("indextts_model"),
                io.Audio.Input("reference_audio",  optional=True, tooltip="Reference audio for voice cloning"),
                TYPE_Audios.Input("reference_audios", optional=True, tooltip="(Optional) Reference audios for voice cloning (Multiple voices are supported)"),
                TYPE_Emotions.Input("emotions", optional=True, tooltip="(Optional) voice emotions"),
                io.String.Input("text", multiline=True, tooltip="Text to synthesize. Supports pause format: -2s- (2 seconds pause), -0.5s- (0.5 seconds pause)"),
                io.Boolean.Input("unload_model", label_on="on", label_off="off", default=False, tooltip="Unload model from VRAM after synthesis"),
                io.Int.Input("seed", default=0, min=0, max=2 ** 32 - 1),
            ],
            outputs=[
                io.Audio.Output(display_name="audio"),
                io.Int.Output(display_name="seed"),
                io.String.Output(display_name="subtitle"),
            ],
        )

    @classmethod
    def execute(cls, indextts_model, text: str, unload_model: bool, seed: int, reference_audio: io.Audio=None, reference_audios=None, emotions=None) -> io.NodeOutput:
        global fingerprint
        # 使用默认参数调用indexTTS2Generate
        return indexTTS2Generate.execute(
            indextts_model=indextts_model,
            text=text,
            unload_model=unload_model,
            do_sample=True,  # 默认值
            temperature=0.8,  # 默认值
            top_p=0.9,  # 默认值
            top_k=30,  # 默认值
            num_beams=3,  # 默认值
            repetition_penalty=10.0,  # 默认值
            length_penalty=0.0,  # 默认值
            max_mel_tokens=1815,  # 默认值
            max_tokens_per_sentence=120,  # 默认值
            seed=seed,
            reference_audio=reference_audio,
            reference_audios=reference_audios,
            emotions=emotions
        )

# 角色情感向量
class indexTTSEmotionVector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy indexTTSEmotionVector",
            display_name="Voice Emotion Vector",
            category="EasyUse/IndexTTS2",
            inputs=[
                io.Audio.Input("reference_audio", tooltip="Reference audio"),
                io.String.Input("voice_name", display_name="Voice name", tooltip="Voice name (If empty, it will be defined as s1, s2..., according to the order of batch combinations)", default=""),
                io.Boolean.Input("use_random", display_name="Random Emotion", label_on="on", label_off="off",
                                 default=False, tooltip="Generate a random emotion sampling"),
                io.Float.Input("Happy", default=0.0, min=0.0, max=1.4, step=0.05, display_mode="slider"),
                io.Float.Input("Angry", default=0.0, min=0.0, max=1.4, step=0.05, display_mode="slider"),
                io.Float.Input("Sad", default=0.0, min=0.0, max=1.4, step=0.05, display_mode="slider"),
                io.Float.Input("Fear", default=0.0, min=0.0, max=1.4, step=0.05, display_mode="slider"),
                io.Float.Input("Hate", default=0.0, min=0.0, max=1.4, step=0.05, display_mode="slider"),
                io.Float.Input("Low", default=0.0, min=0.0, max=1.4, step=0.05, display_mode="slider"),
                io.Float.Input("Surprise", default=0.0, min=0.0, max=1.4, step=0.05, display_mode="slider"),
                io.Float.Input("Neutral", default=0.0, min=0.0, max=1.4, step=0.05, display_mode="slider"),
            ],
            outputs=[
                TYPE_Emotions.Output(display_name="emotions"),
            ]
        )

    @classmethod
    def execute(cls, reference_audio: io.Audio, use_random:bool, voice_name: str, Happy: float, Angry: float, Sad: float, Fear: float, Hate: float, Low: float, Surprise: float, Neutral: float, ) -> io.NodeOutput:
        emo_vector = [Happy, Angry, Sad, Fear, Hate, Low, Surprise, Neutral]
        if voice_name.strip() == "":
            voice_name = None
        return io.NodeOutput([{
            "reference_audio": reference_audio,
            "voice_name": voice_name,
            "emo_mode": EMOTION_MODE.EMO_VECTOR,
            "emo_vector": emo_vector,
            "use_random": use_random,
        }])
    
# 角色情感参考音频
class indexTTSEmotionAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy indexTTSEmotionAudio",
            display_name="Voice Emotion Audio",
            category="EasyUse/IndexTTS2",
            inputs=[
                io.Audio.Input("reference_audio", tooltip="Reference audio"),
                io.String.Input("voice_name", display_name="Voice name", tooltip="Voice name (If empty, it will be defined as s1, s2..., according to the order of batch combinations)", default=""),
                io.Audio.Input("emotion_ref_audio", display_name="Emotion reference audio", tooltip="Reference audio for voice emotions"),
                io.Float.Input("emo_weight", display_name="Emotion Weight", default=0.8, min=0.0, max=1.6, step=0.01, tooltip="Weight for the emotion reference audio" ),
           ],
            outputs=[
                TYPE_Emotions.Output(display_name="emotions"),
            ]
        )

    @classmethod
    def execute(cls, reference_audio: io.Audio, voice_name: str, emotion_ref_audio:io.Audio, emo_weight:float) -> io.NodeOutput:
        if voice_name.strip() == "":
            voice_name = None

        emo_ref = process_audio_input(emotion_ref_audio)
        return io.NodeOutput([{
            "reference_audio": reference_audio,
            "voice_name": voice_name,
            "emo_mode": EMOTION_MODE.EMO_AUDIO,
            "emo_ref_audio": emo_ref,
            "emo_weight": emo_weight
        }])

# 角色情感参考描述
class indexTTSEmotionText(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy indexTTSEmotionText",
            display_name="Voice Emotion Text",
            category="EasyUse/IndexTTS2",
            inputs=[
                io.Audio.Input("reference_audio", tooltip="Reference audio"),
                io.String.Input("voice_name", display_name="Voice name",
                                tooltip="Voice name (If empty, it will be defined as s1, s2..., according to the order of batch combinations)",
                                default=""),
                io.String.Input("description", multiline=True, display_name="Emotion Description",
                              placeholder="Please input emotion description (e.g., happy, angry, sad, etc.)"),
                io.Boolean.Input("use_random", display_name="Random Emotion", label_on="on", label_off="off", default=False,
                                 tooltip="Generate a random emotion sampling"),
            ],
            outputs=[
                TYPE_Emotions.Output(display_name="emotions"),
            ]
        )

    @classmethod
    def execute(cls, reference_audio: io.Audio, voice_name: str, description: str, use_random: bool) -> io.NodeOutput:
        if voice_name.strip() == "":
            voice_name = None

        return io.NodeOutput([{
            "reference_audio": reference_audio,
            "voice_name": voice_name,
            "emo_mode": EMOTION_MODE.SAME_AS_REF if description == '' else EMOTION_MODE.EMO_TEXT,
            "description": description,
            "use_random": use_random,
        }])


# 合并角色情感
class indexTTSEmotionMerge(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = []
        for i in range(1, 11):
            inputs.append(TYPE_Emotions.Input(
                f"emotion{i}", 
                optional=(i > 1), 
                tooltip=f"voice emotion {i}" if i == 1 else f"(Optional) voice emotion {i}"
            ))
        
        return io.Schema(
            node_id="easy indexTTSEmotionMerge",
            display_name="Merge Voice Emotions",
            category="EasyUse/IndexTTS2",
            inputs=inputs,
            outputs=[
                TYPE_Emotions.Output(display_name="emotions"),
            ]
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        merged = []
        for i in range(1, 11):
            emotion = kwargs.get(f"emotion{i}")
            if emotion is not None:
                merged.extend(emotion)
        return io.NodeOutput(merged)

# 合并角色参考音频
class indexTTSAudioMerge(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = []
        for i in range(1, 11):
            inputs.append(io.Audio.Input(
                f"audio{i}", 
                optional=(i > 1), 
                tooltip=f"voice audio {i}" if i == 1 else f"(Optional) voice audio {i}"
            ))
        
        return io.Schema(
            node_id="easy indexTTSAudioMerge",
            display_name="Merge Voice Audios",
            category="EasyUse/IndexTTS2",
            inputs=inputs,
            outputs=[
                TYPE_Audios.Output(display_name="audios"),
            ]
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        merged = []
        for i in range(1, 11):
            audio = kwargs.get(f"audio{i}")
            if audio is not None:
                merged.append(audio)
        return io.NodeOutput(merged)