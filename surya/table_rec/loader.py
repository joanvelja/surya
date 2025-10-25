from typing import Optional

import torch

from surya.common.load import ModelLoader
from surya.logging import get_logger
from surya.settings import settings
from surya.table_rec.model.config import (
    SuryaTableRecConfig,
    SuryaTableRecDecoderConfig,
    DonutSwinTableRecConfig,
)
from surya.table_rec.model.encoderdecoder import TableRecEncoderDecoderModel
from surya.table_rec.processor import SuryaTableRecProcessor

logger = get_logger()


class TableRecModelLoader(ModelLoader):
    def __init__(self, checkpoint: Optional[str] = None):
        super().__init__(checkpoint)

        if self.checkpoint is None:
            self.checkpoint = settings.TABLE_REC_MODEL_CHECKPOINT

    def model(
        self,
        device=settings.TORCH_DEVICE_MODEL,
        dtype=settings.MODEL_DTYPE,
        attention_implementation: Optional[str] = None,
    ) -> TableRecEncoderDecoderModel:
        if device is None:
            device = settings.TORCH_DEVICE_MODEL
        if dtype is None:
            dtype = settings.MODEL_DTYPE

        config = SuryaTableRecConfig.from_pretrained(self.checkpoint)
        decoder_config = config.decoder
        decoder = SuryaTableRecDecoderConfig(**decoder_config)
        config.decoder = decoder

        encoder_config = config.encoder
        encoder = DonutSwinTableRecConfig(**encoder_config)
        config.encoder = encoder

        target_device = device
        target_dtype = dtype

        model = TableRecEncoderDecoderModel.from_pretrained(
            self.checkpoint, config=config, dtype=target_dtype
        )

        try:
            model = model.to(device=target_device)
        except (RuntimeError, NotImplementedError) as exc:
            if str(target_device) != "mps":
                raise

            logger.warning(
                "Failed to move TableRecEncoderDecoderModel to MPS (%s); falling back to CPU float32.",
                exc,
            )
            target_device = "cpu"
            target_dtype = torch.float32
            model = model.to(dtype=target_dtype, device=target_device)

        model = model.eval()

        if settings.COMPILE_ALL or settings.COMPILE_TABLE_REC:
            torch.set_float32_matmul_precision("high")
            torch._dynamo.config.cache_size_limit = 16
            torch._dynamo.config.suppress_errors = False

            logger.info(
                f"Compiling table recognition model {self.checkpoint} on device {target_device} with dtype {target_dtype}"
            )
            compile_args = {"backend": "openxla"} if target_device == "xla" else {}
            model.encoder = torch.compile(model.encoder, **compile_args)
            model.decoder = torch.compile(model.decoder, **compile_args)

        logger.debug(
            f"Loaded table recognition model {self.checkpoint} from {TableRecEncoderDecoderModel.get_local_path(self.checkpoint)} onto device {target_device} with dtype {target_dtype}"
        )
        return model

    def processor(
        self, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE
    ) -> SuryaTableRecProcessor:
        processor = SuryaTableRecProcessor(self.checkpoint)

        processor.token_pad_id = 0
        processor.token_eos_id = 1
        processor.token_bos_id = 1
        processor.token_query_end_id = 4
        return processor
