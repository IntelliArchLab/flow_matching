from transformers import CLIPModel


def test_load_clip_model():
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    assert model.vision_model.config.num_hidden_layers >= 1
