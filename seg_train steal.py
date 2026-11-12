from vajra import Vajra

# Instance segmentation training script
# NOTE: Windows users running as a script should keep the main guard.

def main():
    model = Vajra("vajra-v1-nano-seg")  # baseline segmentation model

    train_results = model.train(
        data=r"F:\material\VayuAI-mainV1\VayuAI-main\datasets\data.yaml",
        epochs=200,
        img_size=1600,
        rect=True,
        workers=0,
        device=0,
        batch=32,
        project="SAMX compare",
        name="VajraV1MerudandaX DCNV2 Conv",
        seed=42,
        patience=30,
        # amp=False,
        # optimizer="AdamW",
        #     lr0=0.0005,
        #     cos_lr=True,


        #     degrees=3.0,
        #     translate=0.05,
        #     scale=0.3,
        #     fliplr=0.5,
        #     hsv_h=0.01,
        #     hsv_s=0.4,
        #     hsv_v=0.25,

        #     mixup=0.0,
        #     copy_paste=0.2,
        #     close_mosaic=25,



    )

    # Validate using the training dataset settings
    metrics = model.val()
    return train_results, metrics


if __name__ == "__main__":
    main()
