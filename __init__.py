from .nodes import *
from typing_extensions import override

class IndexTTS2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            DownloadAndLoadIndexTTSModel,
            indexTTS2Generate,
            indexTTSGenerateSimple,
            indexTTSEmotionAudio,
            indexTTSEmotionVector,
            indexTTSEmotionText,
            indexTTSEmotionMerge,
            indexTTSAudioMerge
        ]

async def comfy_entrypoint() -> IndexTTS2Extension:
    return IndexTTS2Extension()