from vajra import Vajra

# Instance segmentation training script
# NOTE: Windows users running as a script should keep the main guard.

def main():
    model = Vajra("vajra-v1-nano-seg")  # baseline segmentation model

    train_results = model.train(
        data=r"F:\zanbiya_datasets\data.yaml",
        epochs=300,
        img_size=640,
        workers=0,
        device=0,
        batch=32,
        project="zanbiya3",
        name="zanbiya_data",
        seed=42,
        patience=30,


    )

    # Validate using the training dataset settings
    metrics = model.val()
    return train_results, metrics


if __name__ == "__main__":
    main()
