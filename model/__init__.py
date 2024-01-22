from .clip_module import CLIPVisionTower
from .lame import LAME, ImageProjection


def build_lame(args, **kwargs):
    (
        llm_model_id,
        vision_model_id,
        audio_model_id,
    ) = (
        args.llm_model_id,
        args.vision_model_id,
        args.audio_model_id,
    )
    vision_tower = CLIPVisionTower(
        args.mm_vision_select_layer, args.mm_vision_select_feature, vision_model_id
    )
    mm_image_projector = ImageProjection(1, args.mm_image_projection_out_size)

    model = LAME(llm_model_id, vision_tower, mm_image_projector, audio_model_id, **kwargs)

    return model, model.vision_tower.image_processor, model.llm_tokenizer
