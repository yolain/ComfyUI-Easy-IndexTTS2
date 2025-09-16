# ComfyUI Easy IndexTTS2

> \[!免责声明]\
> 本项目及其内容按 “原样 ”提供，不作任何明示或暗示的保证，包括但不限于适销性、特定用途适用性和非侵权保证。在任何情况下，作者或其他版权所有者均不对因本软件或本软件的使用或其他交易而产生、引起或与之相关的任何索赔、损害或其他责任承担责任，无论是合同诉讼、侵权诉讼还是其他诉讼。<br>
本项目严禁用于任何非法目的以及与侵犯版权相关的任何行为, 用户应自行负责确保在使用本软件或发布由本软件生成的内容时，遵守所在司法管辖区的所有适用法律和法规。作者和版权所有者不对用户在其各自所在地违反法律或法规的行为负责。

这个库目前是基于 [ComfyUI_Index_TTS](https://github.com/chenpipi0807/ComfyUI-Index-TTS) 进行微调的 IndexTTS2 魔改版本，底层逻辑和 原库 基本一致，主要改动在于调整了使用流程，
并且额外添加了一些小功能。


## ✨ 主要改动

- [x] 增加 `下载和加载模型` 节点，可选择 HuggingFace 或 modelscope 进行下载
- [x] 增加 `模型卸载` 功能
- [x] 增加 `音色情感参考描述`、`音色情感参考音频`、 `音色情感向量` 节点
- [x] 支持 使用如 `-0.5s-` 的格式在多段对话之间增加停顿时间
- [x] 适配 transformers>=4.56.1 版本
- [x] 节点使用 ComfyUI v3 范式进行编写，如无法启动请更新ComfyUI到较新版本

## 📦 安装与模型路径

1. 克隆项目
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yolain/ComfyUI-Easy-IndexTTS2
```

2. 安装依赖
```bash
cd ComfyUI-Easy-IndexTTS2
..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
```

<details>
<summary><kbd>模型存放路径</kbd></summary>

<br>

```text
与原库路径一致，如果使用的是另一个作者的版本请将模型移动至以下路径，或者使用软链接进行关联
```

1) semantic codec（MaskGCT 语义编码器）
   - 页面：[https://huggingface.co/amphion/MaskGCT/tree/main/semantic_codec](https://huggingface.co/amphion/MaskGCT/tree/main/semantic_codec)
   - 直链：[https://huggingface.co/amphion/MaskGCT/resolve/main/semantic_codec/model.safetensors?download=true](https://huggingface.co/amphion/MaskGCT/resolve/main/semantic_codec/model.safetensors?download=true)
   - 放置：`semantic_codec/model.safetensors`

2) CampPlus 说话人嵌入
   - 页面：[https://huggingface.co/funasr/campplus](https://huggingface.co/funasr/campplus)
   - 直链：[https://huggingface.co/funasr/campplus/resolve/main/campplus_cn_common.bin?download=true](https://huggingface.co/funasr/campplus/resolve/main/campplus_cn_common.bin?download=true)
   - 放置：`campplus_cn_common.bin`

3) Wav2Vec2Bert 特征（facebook/w2v-bert-2.0）
   - 页面：[https://huggingface.co/facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0)
   - 放置：`w2v-bert-2.0/` 整个文件夹（如 `config.json`、`model.safetensors`、`preprocessor_config.json` 等）
   - 若未提前放置，将自动下载到本地缓存：`./ComfyUI/models/IndexTTS-2/hf_cache/`

4) BigVGAN 声码器
   - 依据 `config.yaml` 中 `vocoder.name`（例如 `nvidia/bigvgan_v2_22khz_80band_256x`）
   - 建议提前将对应模型完整缓存到 `bigvgan/` 下

5) 其他本地直读文件（需与 `config.yaml` 一致）：
   - `gpt.pth`（`cfg.gpt_checkpoint`）
   - `s2mel.pth`（`cfg.s2mel_checkpoint`）
   - `bpe.model`（`cfg.dataset.bpe_model`）
   - `wav2vec2bert_stats.pt`（`cfg.w2v_stat`）
   - `qwen0.6bemo4-merge/`（若 `cfg.qwen_emo_path` 指向该目录）
  
6) 基础模型
   - 页面：[TTS2](https://huggingface.co/IndexTeam/IndexTTS-2/tree/main)
   - 放置：`.\ComfyUI\models\IndexTTS-2` 

示例目录结构（部分）：

```text
ComfyUI/models/IndexTTS-2/
│  .gitattributes
│  bpe.model
│  campplus_cn_common.bin
│  config.yaml
│  feat1.pt
│  feat2.pt
│  gpt.pth
│  README.md
│  s2mel.pth
│  wav2vec2bert_stats.pt
│
├─bigvgan
│  └─bigvgan_v2_22khz_80band_256x
│          .gitattributes
│          .gitignore
│          activations.py
│          bigvgan.py
│          bigvgan_discriminator_optimizer.pt
│          bigvgan_discriminator_optimizer_3msteps.pt
│          bigvgan_generator.pt
│          bigvgan_generator_3msteps.pt
│          config.json
│          env.py
│          LICENSE
│          meldataset.py
│          README.md
│          utils.py
│
├─hf_cache
├─qwen0.6bemo4-merge
│      added_tokens.json
│      chat_template.jinja
│      config.json
│      generation_config.json
│      merges.txt
│      model.safetensors
│      Modelfile
│      special_tokens_map.json
│      tokenizer.json
│      tokenizer_config.json
│      vocab.json
│
├─semantic_codec
│      model.safetensors
│
└─w2v-bert-2.0
        .gitattributes
        config.json
        conformer_shaw.pt
        model.safetensors
        preprocessor_config.json
        README.md
```
</details>

## 🥳 IndexTTS Generate 输入参数优先级

`emotions` > `reference_audios` > `reference_audio`

（情感配置 > 参考音频组 > 参考音频）

## 🤯 使用方法

1. 基础单音色克隆

https://github.com/user-attachments/assets/ad18b548-9617-42cd-b086-1bab5e32dbea

2. 以情感为主，进行多段参考克隆

https://github.com/user-attachments/assets/aec59a37-ea53-45c5-9639-5ba8560be378

https://github.com/user-attachments/files/22367059/ComfyUI_00001_.mp3

3. 以音色为主，进行多段参考克隆

https://github.com/user-attachments/assets/dea5238a-face-44e9-ad90-115bc1b8bba9

https://github.com/user-attachments/assets/7a73fbb8-db9d-42d9-8882-2079904675f2


# 来源

- 感谢原作者的开发[ComfyUI-Index-TTS](https://github.com/chenpipi0807/ComfyUI-Index-TTS)
- 基于原始[IndexTTS](https://github.com/index-tts/index-tts)模型
